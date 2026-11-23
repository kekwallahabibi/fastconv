//! Lossless container remuxing: packet/bitstream copy between MP4/MOV and MKV.
//!
//! Uses `containr` as a pure-Rust container library. No codec is decoded or
//! re-encoded. No `ffmpeg`/`ffprobe` subprocess is involved.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

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
fn remux_file(item: &VideoRemuxItem) -> Result<(), PipelineError> {
    let source = item.source();
    let target = item.target();
    match (item.input_format(), item.output_format()) {
        (VideoContainerFormat::Mp4, VideoContainerFormat::Mkv) => mp4_to_mkv(source, target),
        (VideoContainerFormat::Mkv, VideoContainerFormat::Mp4) => mkv_to_mp4(source, target),
        (VideoContainerFormat::Mp4, VideoContainerFormat::Mp4) => mp4_to_mp4(source, target),
        (VideoContainerFormat::Mkv, VideoContainerFormat::Mkv) => mkv_to_mkv(source, target),
    }
}

fn prepare_target(target: &Path) -> Result<PathBuf, PipelineError> {
    if let Some(parent) = target.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    Ok(target.to_path_buf())
}

/// A crash-safe staging target: writes go to a sibling temp file, then rename.
struct StagingTarget {
    temp: PathBuf,
    target: PathBuf,
}

impl StagingTarget {
    fn new(target: &Path, tag: usize) -> Self {
        let name = target.file_name().map_or_else(
            || "remux.mp4".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let temp = target.with_file_name(format!(".{name}.remux-{tag}.tmp"));
        Self {
            temp,
            target: target.to_path_buf(),
        }
    }

    fn commit(&self) -> Result<(), PipelineError> {
        match std::fs::rename(&self.temp, &self.target) {
            Ok(()) => Ok(()),
            Err(source) => {
                let _ = std::fs::remove_file(&self.temp);
                Err(PipelineError::Io {
                    path: self.target.clone(),
                    source,
                })
            }
        }
    }
}

impl Drop for StagingTarget {
    fn drop(&mut self) {
        if self.temp.exists() {
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

fn mp4_to_mkv(source: &Path, target: &Path) -> Result<(), PipelineError> {
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

    let file = File::create(prepare_target(target)?).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let mut output =
        containr::mkv::Muxer::new(file, options).map_err(|err| container_err(target, &err))?;

    let mut ids: Vec<(u32, containr::mkv::TrackId)> = Vec::with_capacity(plans.len());
    for (source_id, plan) in &plans {
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target, &err))?;
        ids.push((*source_id, id));
    }
    let tags = containr::mkv::tags_from_mp4(input.metadata())
        .map_err(|err| container_err(source, &err))?;
    output
        .set_metadata(tags)
        .map_err(|err| container_err(target, &err))?;
    output
        .set_chapters(input.chapters().to_vec())
        .map_err(|err| container_err(target, &err))?;

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
                .map_err(|err| container_err(target, &err))?;
        }
    }
    output.finish().map_err(|err| container_err(target, &err))?;
    Ok(())
}

fn mkv_to_mp4(source: &Path, target: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mkv::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;
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

    prepare_target(target)?;
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(target)
        .map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?;
    let mut output =
        containr::mp4::Muxer::new(file, options).map_err(|err| container_err(target, &err))?;

    let mut plans: HashMap<u32, (containr::mp4::StreamCopy, containr::mp4::TrackId)> =
        HashMap::new();
    for track in input.tracks() {
        let mut packets = read_packets(
            input
                .track_packets(track.id)
                .map_err(|e| container_err(source, &e))?,
            source,
        )?;
        // containr's MP4 edit-list planner keeps a Matroska last block's full
        // BlockDuration and ignores its positive DiscardPadding. Fold the pad
        // into the last duration so the generated edit ends where the source
        // edit did (no trailing samples silently played).
        if let Some(last) = packets.last_mut()
            && let Some(padding) = last.discard_padding
            && padding > 0
            && let Some(duration) = last.duration
        {
            let tick_ns = i128::from(track.timebase.num.max(1));
            let pad_ticks = u64::try_from(((i128::from(padding) + tick_ns / 2) / tick_ns).max(1))
                .unwrap_or(u64::MAX);
            last.duration = Some(duration.saturating_sub(pad_ticks).max(1));
        }
        let plan = containr::mp4::StreamCopy::plan(track, &packets, movie_timescale)
            .map_err(|err| container_err(source, &err))?;
        let id = output
            .add_track(plan.spec.clone())
            .map_err(|err| container_err(target, &err))?;
        plans.insert(track.id, (plan, id));
    }
    if plans.is_empty() {
        return Err(remux_failed(source, "no remuxable tracks in MKV source"));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target, &err))?;

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
            .map_err(|err| container_err(target, &err))?;
    }
    output.finish().map_err(|err| container_err(target, &err))?;
    Ok(())
}

fn mp4_to_mp4(source: &Path, target: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mp4::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;
    let options = containr::mp4::MuxOptions {
        faststart: true,
        ..containr::mp4::MuxOptions::default()
    };
    let movie_timescale = options.movie_timescale;

    prepare_target(target)?;
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(target)
        .map_err(|source| PipelineError::Io {
            path: target.to_path_buf(),
            source,
        })?;
    let mut output =
        containr::mp4::Muxer::new(file, options).map_err(|err| container_err(target, &err))?;

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
            .map_err(|err| container_err(target, &err))?;
        plans.insert(track.id, (plan, id));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target, &err))?;
    output
        .set_chapters(containr::mp4::StreamCopy::chapters(input.chapters()))
        .map_err(|err| container_err(target, &err))?;
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
            .map_err(|err| container_err(target, &err))?;
    }
    output.finish().map_err(|err| container_err(target, &err))?;
    Ok(())
}

fn mkv_to_mkv(source: &Path, target: &Path) -> Result<(), PipelineError> {
    let mut input =
        containr::mkv::Demuxer::open_path(source).map_err(|err| container_err(source, &err))?;
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

    prepare_target(target)?;
    let file = File::create(target).map_err(|source| PipelineError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let mut output =
        containr::mkv::Muxer::new(file, options).map_err(|err| container_err(target, &err))?;
    let mut ids: Vec<(u32, containr::mkv::TrackId)> = Vec::with_capacity(plans.len());
    for (source_id, plan) in &plans {
        ids.push((
            *source_id,
            output
                .add_track(plan.spec.clone())
                .map_err(|err| container_err(target, &err))?,
        ));
    }
    output
        .set_metadata(input.metadata().to_vec())
        .map_err(|err| container_err(target, &err))?;
    output
        .set_chapters(input.chapters().to_vec())
        .map_err(|err| container_err(target, &err))?;
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
                .map_err(|err| container_err(target, &err))?;
        }
    }
    output.finish().map_err(|err| container_err(target, &err))?;
    Ok(())
}

fn remux_with_staging(item: &VideoRemuxItem, tag: usize) -> Result<(), PipelineError> {
    let staging = StagingTarget::new(item.target(), tag);
    // Route the copy to the staging path by building a temporary item.
    let staged = VideoRemuxItem::new(
        item.source().to_path_buf(),
        staging.temp.clone(),
        item.input_format(),
        item.output_format(),
        item.streams().to_vec(),
    );
    remux_file(&staged)?;
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
