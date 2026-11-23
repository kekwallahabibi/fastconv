//! Lossless container remuxing: packet/bitstream copy between MP4/MOV and MKV.
//!
//! Uses `containr` as a pure-Rust container library. No codec is decoded or
//! re-encoded. No `ffmpeg`/`ffprobe` subprocess is involved.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

/// Process-wide serial makes staging paths unique across concurrent invocations.
static STAGING_SERIAL: AtomicUsize = AtomicUsize::new(0);

use containr::Packet;
use fastconv_domain::{
    HardwareBackend, VideoContainerFormat, VideoRemuxItem, VideoRemuxPlan, VideoRemuxReport,
};

use crate::PipelineError;

/// Nanoseconds per Matroska tick used when an exact timestamp plan is not possible.
///
/// A 48 kHz AAC frame and 60 fps video do not land on whole nanoseconds, so
/// `StreamCopy::plan` refuses them. Rounding to 1 µs keeps drift negligible
/// while staying well inside Matroska's tick range.
const ROUNDED_TICK_NS: u64 = 1;

fn remux_failed(path: &Path, reason: impl Into<String>) -> PipelineError {
    PipelineError::RemuxFailed {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn container_err(path: &Path, err: &containr::Error) -> PipelineError {
    remux_failed(path, format!("container error: {err}"))
}

/// Remuxes a single file from one container to the other without re-encoding.
fn remux_file(item: &VideoRemuxItem, target: &File) -> Result<(), PipelineError> {
    let source = item.source();
    match (item.input_format(), item.output_format()) {
        (VideoContainerFormat::Mp4, VideoContainerFormat::Mkv) => {
            mp4_to_mkv(source, target, item.target())
        }
        (VideoContainerFormat::Mkv, VideoContainerFormat::Mp4) => {
            mkv_to_mp4(source, target, item.target())
        }
        (VideoContainerFormat::Mp4, VideoContainerFormat::Mp4) => {
            mp4_to_mp4(source, target, item.target())
        }
        (VideoContainerFormat::Mkv, VideoContainerFormat::Mkv) => {
            mkv_to_mkv(source, target, item.target())
        }
    }
}

/// A crash-safe staging target: the temp file is created exclusively, written
/// through the owned handle, then published with a rename.
struct StagingTarget {
    temp: PathBuf,
    target: PathBuf,
    file: Option<File>,
}

impl StagingTarget {
    fn new(target: &Path, tag: usize) -> Result<Self, PipelineError> {
        if let Some(parent) = target.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let name = target.file_name().map_or_else(
            || "remux.mp4".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let serial = STAGING_SERIAL.fetch_add(1, Ordering::Relaxed);
        let temp = target.with_file_name(format!(
            ".{name}.remux-{}-{serial}-{tag}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(&temp).map_err(|source| PipelineError::Io {
            path: temp.clone(),
            source,
        })?;
        Ok(Self {
            temp,
            target: target.to_path_buf(),
            file: Some(file),
        })
    }

    fn file(&self) -> Result<&File, PipelineError> {
        self.file
            .as_ref()
            .ok_or_else(|| remux_failed(&self.target, "staging file is already committed"))
    }

    fn commit(mut self) -> Result<(), PipelineError> {
        // Close the owned handle before the rename so no writer holds the inode.
        drop(self.file.take());
        std::fs::rename(&self.temp, &self.target).map_err(|source| PipelineError::Io {
            path: self.target.clone(),
            source,
        })
    }
}

impl Drop for StagingTarget {
    fn drop(&mut self) {
        // Only remove the entry this invocation created exclusively.
        if self.file.is_some() {
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

fn read_packets(
    packets: impl Iterator<Item = Result<Packet, containr::Error>>,
    path: &Path,
) -> Result<Vec<Packet>, PipelineError> {
    packets
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| container_err(path, &err))
}

fn mp4_to_mkv(source: &Path, target: &File, target_path: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mp4::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;

    let mut plans: Vec<(u32, containr::mkv::StreamCopy)> = Vec::new();
    for track in input.tracks().iter().filter(|t| !t.chapter_track) {
        let packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        let plan = match containr::mkv::StreamCopy::plan(track, &packets) {
            Ok(plan) => plan,
            Err(_exact) => {
                containr::mkv::StreamCopy::plan_rounded(track, &packets, ROUNDED_TICK_NS)
                    .map_err(|err| container_err(source, &err))?
            }
        };
        plans.push((track.id, plan));
    }
    if plans.is_empty() {
        return Err(remux_failed(source, "no remuxable tracks in MP4 source"));
    }

    let scales: Vec<containr::mkv::StreamCopy> = plans.iter().map(|(_, p)| p.clone()).collect();
    let options = containr::mkv::MkvOptions {
        timestamp_scale: containr::mkv::StreamCopy::timestamp_scale(&scales),
        ..containr::mkv::MkvOptions::default()
    };

    let mut output = containr::mkv::Muxer::new(target, options)
        .map_err(|err| container_err(target_path, &err))?;

    let mut ids: Vec<(u32, containr::mkv::TrackId)> = Vec::with_capacity(plans.len());
    for (source_id, plan) in &plans {
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target_path, &err))?;
        ids.push((*source_id, id));
    }
    let tags = containr::mkv::tags_from_mp4(input.metadata())
        .map_err(|err| container_err(source, &err))?;
    output
        .set_metadata(tags)
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(input.chapters().to_vec())
        .map_err(|err| container_err(target_path, &err))?;

    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some(slot) = ids.iter().position(|(id, _)| *id == packet.track_id) else {
            continue;
        };
        let bytes = input
            .read_packet(&packet)
            .map_err(|err| container_err(source, &err))?;
        if let Some(block) = plans[slot]
            .1
            .packet(&packet, &bytes)
            .map_err(|err| container_err(source, &err))?
        {
            output
                .write_packet(ids[slot].1, &block)
                .map_err(|err| container_err(target_path, &err))?;
        }
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    Ok(())
}

fn mkv_to_mp4(source: &Path, target: &File, target_path: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mkv::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;
    if input.truncated() {
        return Err(remux_failed(source, "truncated MKV input"));
    }
    if !input.attachments().is_empty() {
        return Err(remux_failed(
            source,
            "MKV attachments have no MP4 representation; refusing to drop data",
        ));
    }

    let options = containr::mp4::MuxOptions {
        faststart: true,
        ..containr::mp4::MuxOptions::default()
    };
    let movie_timescale = options.movie_timescale;

    let mut output = containr::mp4::Muxer::new(target, options)
        .map_err(|err| container_err(target_path, &err))?;

    let mut plans: HashMap<u32, (containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        HashMap::new();
    for track in input.tracks() {
        let mut packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        // containr's MP4 edit-list planner ignores DiscardPadding when the last
        // block stores a duration. Only fold the pad into that duration when it
        // still covers a whole frame; some muxers (notably FFmpeg) already trim
        // BlockDuration, and subtracting again would shorten the edit twice.
        let default_duration_ns = track
            .matroska
            .as_ref()
            .and_then(|mkv| mkv.default_duration)
            .map(i128::from);
        if let Some(last) = packets.last_mut()
            && let Some(padding) = last.discard_padding
            && padding > 0
            && let Some(duration) = last.duration
        {
            let tick_ns = i128::from(track.timebase.num.max(1));
            let stored_ns = i128::from(duration) * tick_ns;
            let covers_full_frame =
                default_duration_ns.is_some_and(|frame_ns| stored_ns + tick_ns >= frame_ns);
            if covers_full_frame {
                let pad_ticks =
                    u64::try_from(((i128::from(padding) + tick_ns / 2) / tick_ns).max(1))
                        .unwrap_or(u64::MAX);
                last.duration = Some(duration.saturating_sub(pad_ticks).max(1));
            }
        }
        let plan = containr::mp4::StreamCopy::plan(track, &packets, movie_timescale)
            .map_err(|err| container_err(source, &err))?;
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target_path, &err))?;
        plans.insert(track.id, (plan, id));
    }
    if plans.is_empty() {
        return Err(remux_failed(source, "no remuxable tracks in MKV source"));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target_path, &err))?;

    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some((plan, id)) = plans.get(&packet.track_id) else {
            continue;
        };
        let bytes = input
            .read_packet(&packet)
            .map_err(|err| container_err(source, &err))?;
        let block = plan
            .packet(&packet, &bytes)
            .map_err(|err| container_err(source, &err))?;
        output
            .write_packet(*id, &block)
            .map_err(|err| container_err(target_path, &err))?;
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    Ok(())
}

fn mp4_to_mp4(source: &Path, target: &File, target_path: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mp4::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;
    let options = containr::mp4::MuxOptions {
        faststart: true,
        ..containr::mp4::MuxOptions::default()
    };
    let movie_timescale = options.movie_timescale;

    let mut output = containr::mp4::Muxer::new(target, options)
        .map_err(|err| container_err(target_path, &err))?;

    let mut plans: HashMap<u32, (containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        HashMap::new();
    for track in input.tracks().iter().filter(|t| !t.chapter_track) {
        let packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        let plan = containr::mp4::StreamCopy::plan(track, &packets, movie_timescale)
            .map_err(|err| container_err(source, &err))?;
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target_path, &err))?;
        plans.insert(track.id, (plan, id));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target_path, &err))?;
    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some((plan, id)) = plans.get(&packet.track_id) else {
            continue;
        };
        let bytes = input
            .read_packet(&packet)
            .map_err(|err| container_err(source, &err))?;
        let block = plan
            .packet(&packet, &bytes)
            .map_err(|err| container_err(source, &err))?;
        output
            .write_packet(*id, &block)
            .map_err(|err| container_err(target_path, &err))?;
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    Ok(())
}

fn mkv_to_mkv(source: &Path, target: &File, target_path: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mkv::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;
    if input.truncated() {
        return Err(remux_failed(source, "truncated MKV input"));
    }
    let mut plans: Vec<(u32, containr::mkv::StreamCopy)> = Vec::new();
    for track in input.tracks() {
        let packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        let plan = containr::mkv::StreamCopy::plan(track, &packets)
            .map_err(|err| container_err(source, &err))?;
        plans.push((track.id, plan));
    }
    let scales: Vec<containr::mkv::StreamCopy> = plans.iter().map(|(_, p)| p.clone()).collect();
    let options = containr::mkv::MkvOptions {
        timestamp_scale: containr::mkv::StreamCopy::timestamp_scale(&scales),
        ..containr::mkv::MkvOptions::default()
    };

    let mut output = containr::mkv::Muxer::new(target, options)
        .map_err(|err| container_err(target_path, &err))?;
    let mut ids: Vec<(u32, containr::mkv::TrackId)> = Vec::with_capacity(plans.len());
    for (source_id, plan) in &plans {
        ids.push((
            *source_id,
            output
                .add_track(plan.spec.clone())
                .map_err(|err| container_err(target_path, &err))?,
        ));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target_path, &err))?;
    output
        .set_chapters(input.chapters().to_vec())
        .map_err(|err| container_err(target_path, &err))?;

    // Copy attachments with their UIDs so tags targeting them stay valid.
    let attachments = input.attachments().to_vec();
    for attachment in &attachments {
        let data = input
            .read_attachment(attachment)
            .map_err(|err| container_err(source, &err))?;
        let item = containr::mkv::AttachmentIn {
            name: &attachment.name,
            mime: &attachment.mime,
            description: attachment.description.as_deref(),
            uid: Some(attachment.uid),
            data: &data,
        };
        output
            .attach(&item)
            .map_err(|err| container_err(target_path, &err))?;
    }

    for packet in input.packets() {
        let packet = packet.map_err(|err| container_err(source, &err))?;
        let Some(slot) = ids.iter().position(|(id, _)| *id == packet.track_id) else {
            continue;
        };
        let bytes = input
            .read_packet(&packet)
            .map_err(|err| container_err(source, &err))?;
        if let Some(block) = plans[slot]
            .1
            .packet(&packet, &bytes)
            .map_err(|err| container_err(source, &err))?
        {
            output
                .write_packet(ids[slot].1, &block)
                .map_err(|err| container_err(target_path, &err))?;
        }
    }
    output
        .finish()
        .map_err(|err| container_err(target_path, &err))?;
    Ok(())
}

fn remux_with_staging(item: &VideoRemuxItem, tag: usize) -> Result<(), PipelineError> {
    let staging = StagingTarget::new(item.target(), tag)?;
    remux_file(item, staging.file()?)?;
    staging.commit()
}

/// Executes a validated [`VideoRemuxPlan`] with parallel file-level workers.
///
/// # Errors
///
/// Returns [`PipelineError`] if any container read/write fails. First failure
/// latches and stops remaining workers.
pub fn execute_video_batch(plan: &VideoRemuxPlan) -> Result<VideoRemuxReport, PipelineError> {
    let started = Instant::now();
    let workers = plan.effective_workers().get();
    let total_input: u64 = plan.items().iter().map(|i| input_size(i.source())).sum();

    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<PipelineError>> = Mutex::new(None);
    let tag = AtomicUsize::new(0);

    thread::scope(|scope| -> Result<(), PipelineError> {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let next = &next;
            let failure = &failure;
            let tag = &tag;
            let items = plan.items();
            handles.push(scope.spawn(move || {
                loop {
                    if failure.lock().map_or(true, |guard| guard.is_some()) {
                        break;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= items.len() {
                        break;
                    }
                    let my_tag = tag.fetch_add(1, Ordering::Relaxed);
                    if let Err(err) = remux_with_staging(&items[index], my_tag) {
                        if let Ok(mut guard) = failure.lock()
                            && guard.is_none()
                        {
                            *guard = Some(err);
                        }
                        break;
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| PipelineError::WorkerPanicked)?;
        }
        Ok(())
    })?;

    if let Some(err) = failure
        .into_inner()
        .map_err(|_| PipelineError::WorkerPanicked)?
    {
        return Err(err);
    }

    let total_output: u64 = plan.items().iter().map(|i| input_size(i.target())).sum();
    let duration = started.elapsed();
    let files =
        std::num::NonZeroUsize::new(plan.items().len()).unwrap_or(std::num::NonZeroUsize::MIN);
    let input_bytes =
        std::num::NonZeroU64::new(total_input.max(1)).unwrap_or(std::num::NonZeroU64::MIN);
    let output_bytes =
        std::num::NonZeroU64::new(total_output.max(1)).unwrap_or(std::num::NonZeroU64::MIN);
    Ok(VideoRemuxReport::new(
        files,
        input_bytes,
        output_bytes,
        duration,
        plan.effective_workers(),
        HardwareBackend::Cpu,
    ))
}

fn input_size(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}
