//! Pure-Rust bitstream framing and codec-configuration helpers for container remuxing.
//!
//! All lengths are validated before any allocation. No external command is used.

/// Splits Annex-B bytes into NAL units without their start codes.
fn annex_b_nalus(data: &[u8]) -> Vec<&[u8]> {
    let mut nalus = Vec::new();
    let mut start: Option<usize> = None;
    let mut idx = 0_usize;
    while idx + 2 < data.len() {
        if data[idx] == 0 && data[idx + 1] == 0 && data[idx + 2] == 1 {
            if let Some(from) = start.take() {
                let mut end = idx;
                while end > from && data[end - 1] == 0 {
                    end -= 1;
                }
                if end > from {
                    nalus.push(&data[from..end]);
                }
            }
            start = Some(idx + 3);
            idx += 3;
        } else if data[idx + 2] > 1 {
            // A start code needs `00 00 01` at some offset j. If byte idx+2
            // exceeds 1, no start code can begin at idx, idx+1 or idx+2, so
            // skip all three positions at once. Exact match/slicing semantics
            // are unchanged, only the search stride.
            idx += 3;
        } else {
            idx += 1;
        }
    }
    if let Some(from) = start {
        let mut end = data.len();
        while end > from && data[end - 1] == 0 {
            end -= 1;
        }
        if end > from {
            nalus.push(&data[from..end]);
        }
    }
    nalus
}

fn nalu_type(nalu: &[u8]) -> Option<u8> {
    let header = *nalu.first()?;
    Some(header & 0x1F)
}

fn hevc_nalu_type(nalu: &[u8]) -> Option<u8> {
    let header = *nalu.first()?;
    Some((header >> 1) & 0x3F)
}

fn push_annex_b(out: &mut Vec<u8>, nalu: &[u8]) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nalu);
}

fn be_u16(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_be_bytes(bytes.get(..2)?.try_into().ok()?))
}

fn le_u16(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(..2)?.try_into().ok()?))
}

fn be_u32(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(bytes.get(..4)?.try_into().ok()?))
}

fn le_u32(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

/// Removes H.264/HEVC emulation-prevention bytes (`00 00 03` becomes `00 00`).
fn strip_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0_u8;
    for &byte in data {
        if zeros >= 2 && byte == 0x03 {
            zeros = 0;
            continue;
        }
        out.push(byte);
        zeros = if byte == 0 {
            zeros.saturating_add(1)
        } else {
            0
        };
    }
    out
}

// ---- H.264 ---------------------------------------------------------------

fn avcc_param_sets(avcc: &[u8]) -> Result<(usize, Vec<Vec<u8>>), String> {
    if avcc.len() < 7 || avcc[0] != 1 {
        return Err(format!("invalid avcC record of {} bytes", avcc.len()));
    }
    let length_size = usize::from(avcc[4] & 3)
        .checked_add(1)
        .ok_or("avcC length size overflow")?;
    let count = usize::from(avcc[5] & 0x1F);
    let mut idx = 6_usize;
    let mut sets = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::from(
            be_u16(avcc.get(idx..).ok_or("truncated avcC SPS length")?)
                .ok_or("truncated avcC SPS length")?,
        );
        idx = idx.checked_add(2).ok_or("avcC offset overflow")?;
        let end = idx.checked_add(len).ok_or("avcC SPS overflow")?;
        let nal = avcc.get(idx..end).ok_or("truncated avcC SPS")?;
        sets.push(nal.to_vec());
        idx = end;
    }
    let pps_count = *avcc.get(idx).ok_or("truncated avcC PPS count")?;
    idx = idx.checked_add(1).ok_or("avcC offset overflow")?;
    for _ in 0..pps_count {
        let len = usize::from(
            be_u16(avcc.get(idx..).ok_or("truncated avcC PPS length")?)
                .ok_or("truncated avcC PPS length")?,
        );
        idx = idx.checked_add(2).ok_or("avcC offset overflow")?;
        let end = idx.checked_add(len).ok_or("avcC PPS overflow")?;
        let nal = avcc.get(idx..end).ok_or("truncated avcC PPS")?;
        sets.push(nal.to_vec());
        idx = end;
    }
    Ok((length_size, sets))
}

fn length_prefixed_nalus(sample: &[u8], length_size: usize) -> Result<Vec<&[u8]>, String> {
    let mut nalus = Vec::new();
    let mut idx = 0_usize;
    while idx < sample.len() {
        let size_bytes = sample.get(idx..).ok_or("truncated NALU length")?;
        if size_bytes.len() < length_size {
            return Err("truncated NALU length".to_owned());
        }
        let len = match length_size {
            1 => usize::from(size_bytes[0]),
            2 => usize::from(be_u16(size_bytes).ok_or("truncated NALU length")?),
            3 => {
                let bytes = size_bytes.get(..3).ok_or("truncated NALU length")?;
                usize::from(bytes[0]) << 16 | usize::from(bytes[1]) << 8 | usize::from(bytes[2])
            }
            4 => usize::try_from(be_u32(size_bytes).ok_or("truncated NALU length")?)
                .map_err(|_| "NALU length exceeds address space".to_owned())?,
            _ => return Err(format!("unsupported NALU length size {length_size}")),
        };
        idx = idx
            .checked_add(length_size)
            .ok_or("sample offset overflow")?;
        let end = idx.checked_add(len).ok_or("NALU length overflow")?;
        let nalu = sample.get(idx..end).ok_or("truncated NALU payload")?;
        if !nalu.is_empty() {
            nalus.push(nalu);
        }
        idx = end;
    }
    Ok(nalus)
}

pub(crate) fn avcc_to_annex_b(
    avcc_config: &[u8],
    avcc_sample: &[u8],
    prepend_param_sets: bool,
) -> Result<Vec<u8>, String> {
    let (length_size, params) = avcc_param_sets(avcc_config)?;
    let sample_nalus = length_prefixed_nalus(avcc_sample, length_size)?;
    let mut out = Vec::with_capacity(avcc_sample.len() + params.len() * 8);
    if prepend_param_sets {
        let sps_in_sample = sample_nalus.iter().any(|n| nalu_type(n) == Some(7));
        let pps_in_sample = sample_nalus.iter().any(|n| nalu_type(n) == Some(8));
        for nalu in &params {
            let kind = nalu_type(nalu);
            if (kind == Some(7) && !sps_in_sample) || (kind == Some(8) && !pps_in_sample) {
                push_annex_b(&mut out, nalu);
            }
        }
    }
    for nalu in sample_nalus {
        push_annex_b(&mut out, nalu);
    }
    Ok(out)
}

/// A length-prefixed codec configuration record parsed once per stream.
///
/// `avcC`/`hvcC` parameter sets are immutable for the life of a track; parsing
/// them inside the per-packet conversion loop would repeat the same work for
/// every frame.
#[cfg(feature = "nvenc")]
pub(crate) struct LengthPrefixedConfig {
    length_size: usize,
    params: Vec<Vec<u8>>,
    hevc: bool,
}

#[cfg(feature = "nvenc")]
impl LengthPrefixedConfig {
    pub(crate) fn avcc(config: &[u8]) -> Result<Self, String> {
        let (length_size, params) = avcc_param_sets(config)?;
        Ok(Self {
            length_size,
            params,
            hevc: false,
        })
    }

    pub(crate) fn hvcc(config: &[u8]) -> Result<Self, String> {
        let (length_size, params) = hvcc_param_sets(config)?;
        Ok(Self {
            length_size,
            params,
            hevc: true,
        })
    }
}

/// Writes one AVCC/HVCC sample as Annex-B into reusable `out` and records the
/// `(payload_start, payload_end)` range of every emitted NAL unit in reusable
/// `ranges`. Callers clear both buffers between packets so the output storage
/// is reused (`length_prefixed_nalus` and the HEVC parameter-set check still
/// allocate small per-sample vectors); the byte ranges let access-unit detection
/// skip a second byte-by-byte scan for `00 00 01` start codes.
#[cfg(feature = "nvenc")]
pub(crate) fn length_prefixed_to_annex_b_into(
    config: &LengthPrefixedConfig,
    sample: &[u8],
    prepend_param_sets: bool,
    out: &mut Vec<u8>,
    ranges: &mut Vec<(usize, usize)>,
) -> Result<(), String> {
    let sample_nalus = length_prefixed_nalus(sample, config.length_size)?;
    if prepend_param_sets {
        if config.hevc {
            let present: Vec<u8> = sample_nalus
                .iter()
                .filter_map(|n| hevc_nalu_type(n))
                .collect();
            for nalu in &config.params {
                let Some(kind) = hevc_nalu_type(nalu) else {
                    continue;
                };
                if matches!(kind, 32..=34) && !present.contains(&kind) {
                    push_annex_b(out, nalu);
                    let payload_start = out.len() - nalu.len();
                    ranges.push((payload_start, out.len()));
                }
            }
        } else {
            let sps_in_sample = sample_nalus.iter().any(|n| nalu_type(n) == Some(7));
            let pps_in_sample = sample_nalus.iter().any(|n| nalu_type(n) == Some(8));
            for nalu in &config.params {
                let kind = nalu_type(nalu);
                if (kind == Some(7) && !sps_in_sample) || (kind == Some(8) && !pps_in_sample) {
                    push_annex_b(out, nalu);
                    let payload_start = out.len() - nalu.len();
                    ranges.push((payload_start, out.len()));
                }
            }
        }
    }
    for nalu in sample_nalus {
        push_annex_b(out, nalu);
        let payload_start = out.len() - nalu.len();
        ranges.push((payload_start, out.len()));
    }
    Ok(())
}

pub(crate) fn annex_b_to_avcc(annex_b: &[u8]) -> Vec<u8> {
    annex_b_to_avcc_with_keyframe(annex_b).0
}

/// Converts an Annex-B H.264 access unit to length-prefixed AVCC and reports
/// whether it contains an IDR slice (NALU type 5) in a single start-code pass.
pub(crate) fn annex_b_to_avcc_with_keyframe(annex_b: &[u8]) -> (Vec<u8>, bool) {
    let mut out = Vec::with_capacity(annex_b.len() + 16);
    let mut keyframe = false;
    for nalu in annex_b_nalus(annex_b) {
        let kind = nalu_type(nalu);
        if kind == Some(5) {
            keyframe = true;
        }
        if kind == Some(9) {
            continue;
        }
        let Ok(len) = u32::try_from(nalu.len()) else {
            continue;
        };
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }
    (out, keyframe)
}

pub(crate) fn extract_avcc_from_annex_b(annex_b: &[u8]) -> Option<Vec<u8>> {
    let mut sps: Vec<Vec<u8>> = Vec::new();
    let mut pps: Vec<Vec<u8>> = Vec::new();
    for nalu in annex_b_nalus(annex_b) {
        match nalu_type(nalu) {
            Some(7) if !sps.iter().any(|set| set.as_slice() == nalu) => sps.push(nalu.to_vec()),
            Some(8) if !pps.iter().any(|set| set.as_slice() == nalu) => pps.push(nalu.to_vec()),
            _ => {}
        }
    }
    let first = sps.first()?;
    if sps.is_empty() || pps.is_empty() || sps.len() > 31 || pps.len() > 255 {
        return None;
    }
    let profile = *first.get(1)?;
    let compat = *first.get(2)?;
    let level = *first.get(3)?;
    let mut out = Vec::with_capacity(9 + sps.len() * 8 + pps.len() * 8);
    out.extend_from_slice(&[
        1,
        profile,
        compat,
        level,
        0xFF,
        0xE0 | u8::try_from(sps.len()).ok()?,
    ]);
    for nalu in &sps {
        let len = u16::try_from(nalu.len()).ok()?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }
    out.push(u8::try_from(pps.len()).ok()?);
    for nalu in &pps {
        let len = u16::try_from(nalu.len()).ok()?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }
    Some(out)
}

pub(crate) fn is_h264_keyframe_annex_b(annex_b: &[u8]) -> bool {
    annex_b_nalus(annex_b)
        .iter()
        .any(|n| nalu_type(n) == Some(5))
}

/// Reads the coded dimensions from the first H.264 SPS in an Annex-B stream.
pub(crate) fn h264_dimensions_from_annex_b(data: &[u8]) -> Option<(u16, u16)> {
    parse_h264_sps(data)?.dimensions
}

/// Colour metadata parsed from an H.264/HEVC SPS VUI. `matrix` is `None` when
/// `colour_description_present_flag == 0` so `full_range` is still preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitstreamColor {
    pub(crate) matrix: Option<u8>,
    pub(crate) full_range: bool,
}

/// Colour metadata from the first SPS VUI that signals video signal type.
/// `None` means the source did not signal colour range or matrix.
pub(crate) fn h264_color_from_annex_b(data: &[u8]) -> Option<BitstreamColor> {
    parse_h264_sps(data)?.color
}

struct H264SpsInfo {
    dimensions: Option<(u16, u16)>,
    color: Option<BitstreamColor>,
}

fn parse_h264_sps(data: &[u8]) -> Option<H264SpsInfo> {
    let sps = annex_b_nalus(data)
        .into_iter()
        .find(|nalu| nalu_type(nalu) == Some(7))?;
    let rbsp = strip_emulation_prevention(sps.get(1..)?);
    let mut bits = BitReader::new(&rbsp);
    let profile = u8::try_from(bits.read(8)?).ok()?;
    let _constraints = bits.read(8)?;
    let _level = bits.read(8)?;
    let _sps_id = bits.ue()?;
    let (chroma_format, separate_colour_plane) = h264_profile_extras(&mut bits, profile)?;
    let _log2_max_frame_num = bits.ue()?;
    let pic_order_type = bits.ue()?;
    if pic_order_type == 0 {
        let _log2_pic_order = bits.ue()?;
    } else if pic_order_type == 1 {
        let _always_zero = bits.read(1)?;
        let _offset_non_ref = bits.se()?;
        let _offset_top_bottom = bits.se()?;
        let cycles = bits.ue()?;
        for _ in 0..cycles {
            let _offset = bits.se()?;
        }
    }
    let _max_ref_frames = bits.ue()?;
    let _gaps_allowed = bits.read(1)?;
    let width_mbs = bits.ue()?;
    let height_map_units = bits.ue()?;
    let frame_mbs_only = bits.read(1)?;
    if frame_mbs_only == 0 {
        let _mb_adaptive = bits.read(1)?;
    }
    let _direct_8x8 = bits.read(1)?;
    let mut crop = [0_u32; 4];
    if bits.read(1)? == 1 {
        for value in &mut crop {
            *value = bits.ue()?;
        }
    }
    let chroma_array = if separate_colour_plane {
        0
    } else {
        chroma_format
    };
    let sub_width = u32::from(matches!(chroma_array, 1 | 2)) + 1;
    let sub_height = u32::from(chroma_array == 1) + 1;
    let field_factor = 2_u32 - frame_mbs_only;
    // Crop offsets are expressed in chroma samples; bit depth does not scale
    // the crop unit (only chroma format and field pairing do).
    let width = width_mbs
        .checked_add(1)?
        .checked_mul(16)?
        .checked_sub(crop[0].checked_add(crop[1])?.checked_mul(sub_width)?)?;
    let height = height_map_units
        .checked_add(1)?
        .checked_mul(16)?
        .checked_mul(field_factor)?
        .checked_sub(
            crop[2]
                .checked_add(crop[3])?
                .checked_mul(sub_height)?
                .checked_mul(field_factor)?,
        )?;
    let dimensions = u16::try_from(width).ok().zip(u16::try_from(height).ok());
    // A trailing SPS may end immediately after the crop fields; padding reads
    // as zero there, which simply means "no VUI" and must not lose dimensions.
    let color = if bits.read(1)? == 1 {
        h264_vui_color(&mut bits)
    } else {
        None
    };
    Some(H264SpsInfo { dimensions, color })
}

/// High-profile SPS fields before `log2_max_frame_num_minus4`: chroma format,
/// separate colour planes and the optional scaling lists.
fn h264_profile_extras(bits: &mut BitReader<'_>, profile: u8) -> Option<(u32, bool)> {
    if !matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        return Some((1, false));
    }
    let chroma_format = bits.ue()?;
    let separate_colour_plane = chroma_format == 3 && bits.read(1)? == 1;
    let _luma_depth_minus8 = bits.ue()?;
    let _chroma_depth_minus8 = bits.ue()?;
    let _transform_bypass = bits.read(1)?;
    if bits.read(1)? == 1 {
        let lists = if chroma_format == 3 { 12 } else { 8 };
        for index in 0..lists {
            if bits.read(1)? == 1 {
                let values = if index < 6 { 16 } else { 64 };
                // scaling_list(): a zero nextScale means the rest of the list
                // repeats lastScale and carries no delta_scale bits.
                let mut last_scale = 8_i32;
                let mut next_scale = 8_i32;
                for _ in 0..values {
                    if next_scale != 0 {
                        let delta = bits.se()?;
                        next_scale = last_scale.saturating_add(delta).rem_euclid(256);
                    }
                    last_scale = if next_scale == 0 {
                        last_scale
                    } else {
                        next_scale
                    };
                }
            }
        }
    }
    Some((chroma_format, separate_colour_plane))
}

/// H.264 `vui_parameters` colour fields. Every read is length-checked so a
/// truncated SPS cannot yield invented colour metadata.
fn h264_vui_color(bits: &mut BitReader<'_>) -> Option<BitstreamColor> {
    if bits.read_checked(1)? == 1 && bits.read_checked(8)? == 255 {
        // aspect_ratio_idc == Extended_SAR: two 16-bit values.
        let _sar_width = bits.read_checked(16)?;
        let _sar_height = bits.read_checked(16)?;
    }
    if bits.read_checked(1)? == 1 {
        let _overscan_appropriate = bits.read_checked(1)?;
    }
    if bits.read_checked(1)? == 0 {
        return None;
    }
    let _video_format = bits.read_checked(3)?;
    let full_range = bits.read_checked(1)? == 1;
    let matrix = if bits.read_checked(1)? == 1 {
        let _primaries = bits.read_checked(8)?;
        let _transfer = bits.read_checked(8)?;
        Some(u8::try_from(bits.read_checked(8)?).ok()?)
    } else {
        None
    };
    Some(BitstreamColor { matrix, full_range })
}

// ---- HEVC ----------------------------------------------------------------

fn hvcc_param_sets(hvcc: &[u8]) -> Result<(usize, Vec<Vec<u8>>), String> {
    let length_size = usize::from(*hvcc.get(21).ok_or("truncated hvcC header")? & 3)
        .checked_add(1)
        .ok_or("hvcC length size overflow")?;
    let arrays = usize::from(*hvcc.get(22).ok_or("truncated hvcC array count")?);
    let mut idx = 23_usize;
    let mut sets = Vec::new();
    for _ in 0..arrays {
        let kind = *hvcc.get(idx).ok_or("truncated hvcC array header")? & 0x3F;
        let count = usize::from(
            be_u16(hvcc.get(idx + 1..).ok_or("truncated hvcC array count")?)
                .ok_or("truncated hvcC array count")?,
        );
        idx = idx.checked_add(3).ok_or("hvcC offset overflow")?;
        for _ in 0..count {
            let len = usize::from(
                be_u16(hvcc.get(idx..).ok_or("truncated hvcC NALU length")?)
                    .ok_or("truncated hvcC NALU length")?,
            );
            idx = idx.checked_add(2).ok_or("hvcC offset overflow")?;
            let end = idx.checked_add(len).ok_or("hvcC NALU overflow")?;
            let nal = hvcc.get(idx..end).ok_or("truncated hvcC NALU")?;
            if matches!(kind, 32..=34) {
                sets.push(nal.to_vec());
            }
            idx = end;
        }
    }
    Ok((length_size, sets))
}

pub(crate) fn hvcc_to_annex_b(
    hvcc_config: &[u8],
    hvcc_sample: &[u8],
    prepend_param_sets: bool,
) -> Result<Vec<u8>, String> {
    let (length_size, params) = hvcc_param_sets(hvcc_config)?;
    let sample_nalus = length_prefixed_nalus(hvcc_sample, length_size)?;
    let mut out = Vec::with_capacity(hvcc_sample.len() + params.len() * 8);
    if prepend_param_sets {
        let present: Vec<u8> = sample_nalus
            .iter()
            .filter_map(|n| hevc_nalu_type(n))
            .collect();
        for nalu in &params {
            let kind = hevc_nalu_type(nalu);
            if matches!(kind, Some(32..=34)) && !present.contains(&kind.unwrap_or(0)) {
                push_annex_b(&mut out, nalu);
            }
        }
    }
    for nalu in sample_nalus {
        push_annex_b(&mut out, nalu);
    }
    Ok(out)
}

pub(crate) fn annex_b_to_hvcc(annex_b: &[u8]) -> Vec<u8> {
    annex_b_to_hvcc_with_keyframe(annex_b).0
}

/// Converts an Annex-B HEVC access unit to length-prefixed HVCC and reports
/// whether it contains an IRAP slice (NALU types 16..=21) in a single pass.
pub(crate) fn annex_b_to_hvcc_with_keyframe(annex_b: &[u8]) -> (Vec<u8>, bool) {
    let mut out = Vec::with_capacity(annex_b.len() + 16);
    let mut keyframe = false;
    for nalu in annex_b_nalus(annex_b) {
        let kind = hevc_nalu_type(nalu);
        if matches!(kind, Some(16..=21)) {
            keyframe = true;
        }
        if kind == Some(35) {
            continue;
        }
        let Ok(len) = u32::try_from(nalu.len()) else {
            continue;
        };
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }
    (out, keyframe)
}

pub(crate) fn extract_hvcc_from_annex_b(annex_b: &[u8]) -> Option<Vec<u8>> {
    let mut vps = Vec::new();
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for nalu in annex_b_nalus(annex_b) {
        match hevc_nalu_type(nalu) {
            Some(32) => vps.push(nalu.to_vec()),
            Some(33) => sps.push(nalu.to_vec()),
            Some(34) => pps.push(nalu.to_vec()),
            _ => {}
        }
    }
    if sps.is_empty() || pps.is_empty() {
        return None;
    }
    let first_sps = sps.first()?;
    // The two-byte NAL header is not part of the RBSP. The first RBSP byte
    // holds sps_video_parameter_set_id / max_sub_layers / nesting flag;
    // profile_tier_level follows it and ends with general_level_idc.
    let rbsp = strip_emulation_prevention(first_sps.get(2..).unwrap_or(&[]));
    let mut profile_tier_level = [0_u8; 12];
    if let Some(bytes) = rbsp.get(1..) {
        let copied = bytes.len().min(profile_tier_level.len());
        profile_tier_level[..copied].copy_from_slice(&bytes[..copied]);
    }
    let mut out = Vec::with_capacity(23 + vps.len() * 8 + sps.len() * 8 + pps.len() * 8);
    out.push(1);
    out.extend_from_slice(&profile_tier_level);
    out.extend_from_slice(&[0xF0, 0x00, 0xFC, 0xFC, 0xF8, 0xF8, 0x00, 0x00, 0x03, 3]);
    for (kind, sets) in [(32_u8, &vps), (33, &sps), (34, &pps)] {
        let count = u16::try_from(sets.len()).ok()?;
        out.push(kind);
        out.extend_from_slice(&count.to_be_bytes());
        for nalu in sets {
            let len = u16::try_from(nalu.len()).ok()?;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(nalu);
        }
    }
    Some(out)
}

/// Colour metadata from the first HEVC SPS VUI. `None` means the source did
/// not signal colour range or matrix.
pub(crate) fn hevc_color_from_annex_b(data: &[u8]) -> Option<BitstreamColor> {
    let sps = annex_b_nalus(data)
        .into_iter()
        .find(|nalu| hevc_nalu_type(nalu) == Some(33))?;
    let rbsp = strip_emulation_prevention(sps.get(2..)?);
    let parsed = rusty_h265::ps::parse_sps(&rbsp).ok()?;
    let vui = parsed.vui?;
    // `rusty_h265::ps::Vui` leaves `matrix_coeffs` at 0 when colour_description
    // is absent; preserve `video_full_range_flag` even without a matrix.
    if vui.matrix_coeffs == 0 && !vui.video_full_range_flag {
        return None;
    }
    Some(BitstreamColor {
        matrix: (vui.matrix_coeffs != 0).then_some(vui.matrix_coeffs),
        full_range: vui.video_full_range_flag,
    })
}

pub(crate) fn is_hevc_keyframe_annex_b(annex_b: &[u8]) -> bool {
    annex_b_nalus(annex_b)
        .iter()
        .any(|n| matches!(hevc_nalu_type(n), Some(16..=21)))
}

/// Reads the luma dimensions from the first HEVC SPS in an Annex-B stream.
pub(crate) fn hevc_dimensions_from_annex_b(data: &[u8]) -> Option<(u16, u16)> {
    let sps = annex_b_nalus(data)
        .into_iter()
        .find(|nalu| hevc_nalu_type(nalu) == Some(33))?;
    let rbsp = strip_emulation_prevention(sps.get(2..)?);
    let mut bits = BitReader::new(&rbsp);
    let _vps_id = bits.read(4)?;
    let max_sub_layers_minus1 = bits.read(3)?;
    let _nesting = bits.read(1)?;
    // profile_tier_level: profile_space/tier/profile_idc, 32 compatibility
    // flags, 48 constraint bits and the level byte.
    let _profile = bits.read(8)?;
    let _compat = bits.read(32)?;
    let _constraints_high = bits.read(32)?;
    let _constraints_low = bits.read(16)?;
    let _level = bits.read(8)?;
    // ITU-T H.265 §7.3.3 profile_tier_level(1, maxNumSubLayersMinus1): all
    // presence flags come first, then the reserved bits, then one pass over
    // the sublayers for their profile and level payloads in sublayer order.
    let mut sub_layers = Vec::with_capacity(usize::try_from(max_sub_layers_minus1).unwrap_or(0));
    for _ in 0..max_sub_layers_minus1 {
        sub_layers.push((bits.read(1)?, bits.read(1)?));
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            let _reserved = bits.read(2)?;
        }
    }
    for (sub_layer_profile_present_flag, sub_layer_level_present_flag) in &sub_layers {
        if *sub_layer_profile_present_flag == 1 {
            let _sub_profile = bits.read(8)?;
            let _sub_compat = bits.read(32)?;
            let _sub_constraints_high = bits.read(32)?;
            let _sub_constraints_low = bits.read(16)?;
        }
        if *sub_layer_level_present_flag == 1 {
            let _sub_level_idc = bits.read(8)?;
        }
    }
    let _sps_id = bits.ue()?;
    let chroma_format = bits.ue()?;
    if chroma_format == 3 {
        let _separate_colour_plane = bits.read(1)?;
    }
    let width = bits.ue()?;
    let height = bits.ue()?;
    let sub_width = u32::from(matches!(chroma_format, 1 | 2)) + 1;
    let sub_height = u32::from(chroma_format == 1) + 1;
    let mut crop = [0_u32; 4];
    if bits.read(1)? == 1 {
        for value in &mut crop {
            *value = bits.ue()?;
        }
    }
    let width = width.checked_sub(crop[0].checked_add(crop[1])?.checked_mul(sub_width)?)?;
    let height = height.checked_sub(crop[2].checked_add(crop[3])?.checked_mul(sub_height)?)?;
    u16::try_from(width).ok().zip(u16::try_from(height).ok())
}

// ---- AV1 -----------------------------------------------------------------

struct Obu {
    kind: u8,
    bytes: Vec<u8>,
    payload_start: usize,
}

fn obus(data: &[u8]) -> Vec<Obu> {
    let mut result = Vec::new();
    let mut idx = 0_usize;
    while idx < data.len() {
        let header = data[idx];
        if header & 0x80 != 0 {
            break;
        }
        let kind = (header >> 3) & 0x0F;
        let has_size = header & 0x02 != 0;
        let has_ext = header & 0x04 != 0;
        let mut cursor = idx.checked_add(1).unwrap_or(data.len());
        if has_ext {
            cursor = cursor.checked_add(1).unwrap_or(data.len());
        }
        if cursor > data.len() {
            break;
        }
        if has_size {
            let mut size = 0_usize;
            let mut shift = 0_u32;
            loop {
                let byte = match data.get(cursor) {
                    Some(byte) => *byte,
                    None => return result,
                };
                cursor = cursor.saturating_add(1);
                let part = usize::from(byte & 0x7F).checked_shl(shift).unwrap_or(0);
                size = size.saturating_add(part);
                shift = shift.saturating_add(7);
                if byte & 0x80 == 0 || shift > 28 {
                    break;
                }
            }
            let end = cursor.saturating_add(size).min(data.len());
            if end < cursor {
                break;
            }
            result.push(Obu {
                kind,
                bytes: data[idx..end].to_vec(),
                payload_start: cursor - idx,
            });
            idx = end;
        } else {
            result.push(Obu {
                kind,
                bytes: data[idx..].to_vec(),
                payload_start: cursor - idx,
            });
            break;
        }
    }
    result
}

pub(crate) fn ensure_av1_temporal_delimiter(obu_data: &[u8]) -> Vec<u8> {
    let first_kind = obus(obu_data).first().map(|obu| obu.kind);
    if first_kind == Some(2) {
        return obu_data.to_vec();
    }
    let mut out = Vec::with_capacity(obu_data.len() + 2);
    out.extend_from_slice(&[0x12, 0x00]);
    out.extend_from_slice(obu_data);
    out
}

/// MSB-first bit reader for codec headers. Bits past the end read as zero so a
/// truncated header still yields a usable (if conservative) config.
struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    fn read(&mut self, count: usize) -> Option<u32> {
        if count > 32 {
            return None;
        }
        let mut value = 0_u32;
        for _ in 0..count {
            let byte = self.bytes.get(self.bit / 8).copied().unwrap_or(0);
            let shift = 7 - (self.bit % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.bit = self.bit.saturating_add(1);
        }
        Some(value)
    }

    /// `read` pads past the end with zeros; color configuration must instead
    /// fail when a header is truncated so no synthetic metadata is derived.
    fn read_checked(&mut self, count: usize) -> Option<u32> {
        let end = self.bit.checked_add(count)?;
        if end > self.bytes.len().saturating_mul(8) {
            return None;
        }
        self.read(count)
    }

    fn skip_uvlc(&mut self) -> Option<()> {
        let mut zeros = 0_usize;
        while self.read(1)? == 0 {
            zeros = zeros.saturating_add(1);
            if zeros > 32 {
                return None;
            }
        }
        for _ in 0..zeros {
            let _bit = self.read(1)?;
        }
        Some(())
    }

    /// Unsigned Exp-Golomb code (`ue(v)`).
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0_u32;
        while self.read(1)? == 0 {
            zeros = zeros.checked_add(1)?;
            if zeros > 31 {
                return None;
            }
        }
        let suffix = self.read(zeros as usize)?;
        (1_u32.checked_shl(zeros)?)
            .checked_add(suffix)?
            .checked_sub(1)
    }

    /// Signed Exp-Golomb code (`se(v)`).
    fn se(&mut self) -> Option<i32> {
        let code = self.ue()?;
        let magnitude = i64::from(code.saturating_add(1) / 2);
        let signed = if code % 2 == 0 { -magnitude } else { magnitude };
        i32::try_from(signed).ok()
    }
}

pub(crate) fn extract_av1c_from_obu(obu_data: &[u8]) -> Option<Vec<u8>> {
    let seq = obus(obu_data).into_iter().find(|obu| obu.kind == 1)?;
    let payload = seq.bytes.get(seq.payload_start..)?;
    if payload.is_empty() {
        return None;
    }
    let mut bits = BitReader::new(payload);
    let seq_profile = bits.read(3)?;
    let _still_picture = bits.read(1)?;
    let reduced_still_picture_header = bits.read(1)? == 1;
    let (seq_level_idx_0, seq_tier_0) = if reduced_still_picture_header {
        (bits.read(5)?, 0)
    } else {
        let timing_info_present = bits.read(1)? == 1;
        if timing_info_present {
            let _num_units_in_display_tick = bits.read(32)?;
            let _time_scale = bits.read(32)?;
            if bits.read(1)? == 1 {
                bits.skip_uvlc()?;
            }
            if bits.read(1)? == 1 {
                let _buffer_delay_length_minus_1 = bits.read(5)?;
                let _num_units_in_decoding_tick = bits.read(32)?;
                let _queue_depth_length_minus_1 = bits.read(5)?;
                let _buffer_removal_time_length_minus_1 = bits.read(5)?;
            }
        }
        let _initial_display_delay_present = bits.read(1)?;
        let _operating_points_cnt_minus_1 = bits.read(5)?;
        let _operating_point_idc_0 = bits.read(12)?;
        let level = bits.read(5)?;
        let tier = if level > 7 { bits.read(1)? } else { 0 };
        (level, tier)
    };
    let mut out = Vec::with_capacity(4 + seq.bytes.len());
    out.push(0x81);
    out.push(u8::try_from((seq_profile << 5) | seq_level_idx_0).unwrap_or(0));
    // 4:2:0 8-bit: high_bitdepth=0, twelve_bit=0, monochrome=0,
    // chroma_subsampling_x=1, chroma_subsampling_y=1, sample_position=0.
    out.push(u8::try_from(((seq_tier_0 & 1) << 7) | 0x0C).unwrap_or(0x0C));
    out.push(0);
    out.extend_from_slice(&seq.bytes);
    Some(out)
}

pub(crate) fn is_av1_keyframe(obu_data: &[u8]) -> bool {
    let frames = obus(obu_data);
    // With reduced_still_picture_header the frame header omits both fields and
    // every frame is an intra-only still picture.
    let reduced_still = frames
        .iter()
        .find(|obu| obu.kind == 1)
        .and_then(|seq| {
            let payload = seq.bytes.get(seq.payload_start..)?;
            let mut bits = BitReader::new(payload);
            let _profile = bits.read(3)?;
            let _still_picture = bits.read(1)?;
            bits.read(1)
        })
        .unwrap_or(0)
        == 1;
    frames.iter().any(|obu| {
        if obu.kind != 3 && obu.kind != 6 {
            return false;
        }
        if reduced_still {
            return true;
        }
        let Some(payload) = obu.bytes.get(obu.payload_start..) else {
            return false;
        };
        let mut bits = BitReader::new(payload);
        let Some(show_existing) = bits.read(1) else {
            return false;
        };
        if show_existing == 1 {
            return false;
        }
        bits.read(2) == Some(0)
    })
}

/// Color configuration derived from a VP9 keyframe header (VP9 §6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Vp9KeyframeColor {
    pub(crate) profile: u8,
    pub(crate) bit_depth: u8,
    pub(crate) chroma_subsampling: u8,
    pub(crate) full_range: bool,
    pub(crate) colour_primaries: u8,
    pub(crate) transfer: u8,
    pub(crate) matrix: u8,
}

/// Derives the mandatory `color_config()` from a VP9 keyframe header. Inter
/// frames inherit these values from an earlier keyframe and cannot establish
/// them on their own. Every read is checked against the byte length so a
/// truncated header never yields zero-padded metadata.
pub(crate) fn vp9_color_from_keyframe(frame: &[u8]) -> Option<Vp9KeyframeColor> {
    let mut bits = BitReader::new(frame);
    if bits.read_checked(2)? != 2 {
        return None;
    }
    // profile_low_bit precedes profile_high_bit in the VP9 header.
    let profile = bits.read_checked(1)? | (bits.read_checked(1)? << 1);
    if profile == 3 && bits.read_checked(1)? != 0 {
        return None;
    }
    // show_existing_frame carries no color config; only key frames do.
    if bits.read_checked(1)? != 0 || bits.read_checked(1)? != 0 {
        return None;
    }
    let _show_frame = bits.read_checked(1)?;
    let _error_resilient_mode = bits.read_checked(1)?;
    if bits.read_checked(24)? != 0x0049_8342 {
        return None;
    }
    let bit_depth = if profile >= 2 {
        if bits.read_checked(1)? == 1 { 12 } else { 10 }
    } else {
        8
    };
    let color_space = bits.read_checked(3)?;
    let (chroma_subsampling, full_range, colour_primaries, transfer, matrix) = match color_space {
        // CS_RGB exists only for profiles 1 and 3 and implies 4:4:4 full range
        // with sRGB primaries and identity matrix.
        7 => {
            if profile == 0 || profile == 2 {
                return None;
            }
            if bits.read_checked(1)? != 0 {
                return None;
            }
            (3, true, 1, 13, 0)
        }
        0..=5 => {
            let full_range = bits.read_checked(1)? == 1;
            let chroma_subsampling = if profile == 1 || profile == 3 {
                let subsampling_x = bits.read_checked(1)?;
                let subsampling_y = bits.read_checked(1)?;
                if bits.read_checked(1)? != 0 {
                    return None;
                }
                match (subsampling_x, subsampling_y) {
                    (0, 0) => 3,
                    (1, 0) => 2,
                    _ => return None,
                }
            } else {
                0
            };
            let primaries = match color_space {
                0 => (2, 2, 2),
                1 | 3 => (6, 6, 6),
                2 => (1, 1, 1),
                4 => (7, 7, 7),
                5 => (9, if bit_depth == 12 { 15 } else { 14 }, 9),
                _ => return None,
            };
            (
                chroma_subsampling,
                full_range,
                primaries.0,
                primaries.1,
                primaries.2,
            )
        }
        _ => return None,
    };
    Some(Vp9KeyframeColor {
        profile: u8::try_from(profile).ok()?,
        bit_depth,
        chroma_subsampling,
        full_range,
        colour_primaries,
        transfer,
        matrix,
    })
}

// ---- AAC -----------------------------------------------------------------

pub(crate) fn parse_audio_specific_config(asc: &[u8]) -> Option<(u8, u8, u8)> {
    let first = *asc.first()?;
    let second = *asc.get(1)?;
    let object_type = (first >> 3) & 0x1F;
    let freq_index = ((first & 0x07) << 1) | (second >> 7);
    let channels = (second >> 3) & 0x0F;
    Some((object_type, freq_index, channels))
}

pub(crate) fn wrap_adts(asc: &[u8], raw_aac: &[u8]) -> Result<Vec<u8>, String> {
    let (aot, freq_index, channels) =
        parse_audio_specific_config(asc).ok_or("invalid AudioSpecificConfig")?;
    if !(1..=4).contains(&aot) || freq_index > 14 || !(1..=7).contains(&channels) {
        return Err(format!(
            "unsupported AudioSpecificConfig ({aot}, {freq_index}, {channels})"
        ));
    }
    let frame_len = raw_aac
        .len()
        .checked_add(7)
        .ok_or("ADTS frame length overflow")?;
    if frame_len > 0x1FFF {
        return Err("ADTS frame exceeds 13-bit length limit".to_owned());
    }
    let frame_len_u16 = u16::try_from(frame_len).map_err(|_| "ADTS frame too large".to_owned())?;
    let profile = aot - 1;
    let mut out = Vec::with_capacity(frame_len);
    out.push(0xFF);
    out.push(0xF1);
    out.push((profile << 6) | (freq_index << 2) | (channels >> 2));
    // ADTS frame_length is 13 bits: top 2 bits go into byte 3 bits 1..=0.
    out.push((channels << 6) | (u8::try_from(frame_len_u16 >> 11).unwrap_or(0) & 0x03));
    out.push(u8::try_from((frame_len_u16 >> 3) & 0xFF).unwrap_or(0));
    out.push(u8::try_from(((frame_len_u16 & 0x07) << 5) | 0x1F).unwrap_or(0x1F));
    out.push(0xFC);
    out.extend_from_slice(raw_aac);
    Ok(out)
}

pub(crate) fn unwrap_adts(adts_stream: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut idx = 0_usize;
    while idx + 7 <= adts_stream.len() {
        if adts_stream[idx] != 0xFF || adts_stream[idx + 1] & 0xF6 != 0xF0 {
            idx += 1;
            continue;
        }
        let protection_absent = adts_stream[idx + 1] & 0x01 != 0;
        let header_len = if protection_absent { 7 } else { 9 };
        if idx + header_len > adts_stream.len() {
            break;
        }
        let profile = adts_stream[idx + 2] >> 6;
        let freq_index = (adts_stream[idx + 2] >> 2) & 0x0F;
        let channels = ((adts_stream[idx + 2] & 0x01) << 2) | (adts_stream[idx + 3] >> 6);
        let frame_len = (usize::from(adts_stream[idx + 3] & 0x03) << 11)
            | (usize::from(adts_stream[idx + 4]) << 3)
            | usize::from(adts_stream[idx + 5] >> 5);
        if frame_len < header_len {
            idx += 1;
            continue;
        }
        let end = match idx.checked_add(frame_len) {
            Some(end) if end <= adts_stream.len() => end,
            _ => break,
        };
        let payload = adts_stream[idx + header_len..end].to_vec();
        let asc = vec![
            ((profile + 1) << 3) | (freq_index >> 1),
            ((freq_index & 1) << 7) | (channels << 3),
        ];
        out.push((payload, asc));
        idx = end;
    }
    out
}

// ---- Opus ----------------------------------------------------------------

pub(crate) fn dops_to_opus_head(dops: &[u8]) -> Result<Vec<u8>, String> {
    // A dOps body has eleven mandatory bytes; the channel-mapping count and
    // table follow only when the mapping family is non-zero. Unlike OpusHead,
    // dOps uses version 0 and big-endian multi-byte fields.
    if dops.len() < 11 {
        return Err(format!("dOps record of {} bytes is too short", dops.len()));
    }
    if dops[0] != 0 {
        return Err(format!("unsupported dOps version {}", dops[0]));
    }
    let channels = dops[1];
    if channels == 0 {
        return Err("dOps declares zero channels".to_owned());
    }
    let pre_skip =
        be_u16(dops.get(2..).ok_or("truncated dOps pre-skip")?).ok_or("truncated dOps pre-skip")?;
    let input_rate =
        be_u32(dops.get(4..).ok_or("truncated dOps rate")?).ok_or("truncated dOps rate")?;
    let gain = be_u16(dops.get(8..).ok_or("truncated dOps gain")?).ok_or("truncated dOps gain")?;
    let mapping_family = dops[10];
    let mut out = Vec::with_capacity(19);
    out.extend_from_slice(b"OpusHead");
    out.push(1_u8);
    out.push(channels);
    out.extend_from_slice(&pre_skip.to_le_bytes());
    out.extend_from_slice(&input_rate.to_le_bytes());
    out.extend_from_slice(&gain.to_le_bytes());
    out.push(mapping_family);
    if mapping_family != 0 {
        // dOps bytes 11.. are [stream_count, coupled_count, mapping[..]],
        // identical in layout to OpusHead bytes 19...
        out.extend_from_slice(&dops[11..]);
    }
    Ok(out)
}

pub(crate) fn opus_head_to_dops(head: &[u8]) -> Result<Vec<u8>, String> {
    if head.len() < 19 || &head[..8] != b"OpusHead" {
        return Err("invalid OpusHead".to_owned());
    }
    let channels = head[9];
    if channels == 0 {
        return Err("OpusHead declares zero channels".to_owned());
    }
    let pre_skip = le_u16(head.get(10..).ok_or("truncated OpusHead pre-skip")?)
        .ok_or("truncated OpusHead pre-skip")?;
    let input_rate = le_u32(head.get(12..).ok_or("truncated OpusHead rate")?)
        .ok_or("truncated OpusHead rate")?;
    let gain = le_u16(head.get(16..).ok_or("truncated OpusHead gain")?)
        .ok_or("truncated OpusHead gain")?;
    let mapping_family = head[18];
    let mut out = Vec::with_capacity(19);
    out.push(0_u8);
    out.push(channels);
    out.extend_from_slice(&pre_skip.to_be_bytes());
    out.extend_from_slice(&input_rate.to_be_bytes());
    out.extend_from_slice(&gain.to_be_bytes());
    out.push(mapping_family);
    if mapping_family != 0 {
        out.extend_from_slice(&head[19..]);
    }
    Ok(out)
}

pub(crate) fn default_opus_head(channels: u16, pre_skip: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(19);
    out.extend_from_slice(b"OpusHead");
    out.push(1);
    out.push(u8::try_from(channels.clamp(1, 255)).unwrap_or(2));
    out.extend_from_slice(&pre_skip.to_le_bytes());
    out.extend_from_slice(&48_000_u32.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.push(0);
    out
}

pub(crate) fn wrap_ts_opus(raw_opus: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw_opus.len() + 4);
    out.extend_from_slice(&[0x7F, 0xE0]);
    let full = raw_opus.len() / 255;
    out.extend(std::iter::repeat_n(0xFF_u8, full));
    out.push(u8::try_from(raw_opus.len() % 255).unwrap_or(0));
    out.extend_from_slice(raw_opus);
    out
}

pub(crate) fn unwrap_ts_opus(pes_payload: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut idx = 0_usize;
    while idx + 3 <= pes_payload.len() {
        if pes_payload[idx] != 0x7F || pes_payload[idx + 1] != 0xE0 {
            idx += 1;
            continue;
        }
        idx += 2;
        let mut size = 0_usize;
        while idx < pes_payload.len() && pes_payload[idx] == 0xFF {
            size = size.saturating_add(255);
            idx += 1;
        }
        let Some(last) = pes_payload.get(idx).copied() else {
            break;
        };
        idx += 1;
        size = size.saturating_add(usize::from(last));
        let Some(end) = idx.checked_add(size) else {
            break;
        };
        if end > pes_payload.len() {
            break;
        }
        out.push(pes_payload[idx..end].to_vec());
        idx = end;
    }
    out
}

/// Decodes the coded duration of an Opus packet from its RFC 6716 TOC byte.
///
/// Returns the number of 48 kHz samples, rejecting malformed packets and
/// durations above the 120 ms (5760 samples) packet cap.
pub(crate) fn opus_packet_samples_48k(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let frame_samples: u32 = match config {
        0..=11 => match config % 4 {
            0 => 480,
            1 => 960,
            2 => 1920,
            _ => 2880,
        },
        12..=15 => {
            if config % 2 == 0 {
                480
            } else {
                960
            }
        }
        16..=31 => match config % 4 {
            0 => 120,
            1 => 240,
            2 => 480,
            _ => 960,
        },
        _ => return None,
    };
    let frame_count = match toc & 0x03 {
        0 => 1,
        1 | 2 => 2,
        _ => {
            let n = u32::from(*packet.get(1)? & 0x3F);
            if n == 0 {
                return None;
            }
            n
        }
    };
    let total = frame_samples.checked_mul(frame_count)?;
    if total == 0 || total > 5760 {
        None
    } else {
        Some(total)
    }
}

// ---- Vorbis (Matroska Xiph lacing) ----------------------------------------

/// Cap on a Vorbis header set stored in Matroska `CodecPrivate`. A hostile
/// track must never make us allocate an unbounded buffer while unpacking.
const MAX_XIPH_VORBIS_HEADERS: usize = 1_048_576;

/// The three Xiph-laced Vorbis packets borrowed from a `CodecPrivate` blob.
type VorbisHeaders<'a> = (&'a [u8], &'a [u8], &'a [u8]);

fn is_vorbis_header(packet: &[u8], kind: u8) -> bool {
    packet.len() >= 7 && packet[0] == kind && &packet[1..7] == b"vorbis"
}

/// Validates a Vorbis identification header and returns
/// `(channels, sample_rate, bs0, bs1)`.
///
/// All fields are read with checked offsets; every Vorbis I constraint that
/// affects decoding is enforced so a malformed header is rejected before the
/// codec is instantiated.
pub(crate) fn vorbis_ident_info(ident: &[u8]) -> Result<(u16, u32, u64, u64), String> {
    if !is_vorbis_header(ident, 0x01) || ident.len() < 30 {
        return Err("invalid Vorbis identification header".to_owned());
    }
    let version = le_u32(ident.get(7..11).ok_or("truncated Vorbis version")?)
        .ok_or("truncated Vorbis version")?;
    if version != 0 {
        return Err(format!("unsupported Vorbis version {version}"));
    }
    let channels = u16::from(ident[11]);
    if channels == 0 {
        return Err("Vorbis identification header declares zero channels".to_owned());
    }
    let sample_rate =
        le_u32(ident.get(12..16).ok_or("truncated Vorbis rate")?).ok_or("truncated Vorbis rate")?;
    if sample_rate == 0 {
        return Err("Vorbis identification header declares zero sample rate".to_owned());
    }
    let blocksize = ident[28];
    let bs0 = u64::from(blocksize & 0x0F);
    let bs1 = u64::from(blocksize >> 4);
    if !(6..=13).contains(&bs0) || !(6..=13).contains(&bs1) || bs0 > bs1 {
        return Err(format!(
            "invalid Vorbis block sizes {bs0}/{bs1} (expected 6..=13 with bs0 <= bs1)"
        ));
    }
    if ident[29] & 1 == 0 {
        return Err("Vorbis identification header has no framing bit".to_owned());
    }
    Ok((channels, sample_rate, bs0, bs1))
}

/// Appends a Matroska Xiph-laced length: `255` bytes are emitted for each
/// full 255-unit chunk, followed by a final byte `< 255`.
fn push_xiph_length(out: &mut Vec<u8>, len: usize) -> Result<(), String> {
    let mut remaining = len;
    while remaining >= 255 {
        out.push(0xFF);
        remaining = remaining.checked_sub(255).ok_or("Xiph length underflow")?;
    }
    out.push(u8::try_from(remaining).map_err(|_| "Xiph length overflow".to_owned())?);
    Ok(())
}

/// Reads one Xiph-laced length, advancing `idx`. A stream that ends on `0xFF`
/// is truncated and rejected instead of being silently accepted.
fn read_xiph_length(data: &[u8], idx: &mut usize) -> Result<usize, String> {
    let mut len = 0_usize;
    loop {
        let byte = *data.get(*idx).ok_or("truncated Xiph lacing")?;
        *idx = idx.checked_add(1).ok_or("Xiph lacing overflow")?;
        if byte == 0xFF {
            len = len.checked_add(255).ok_or("Xiph length overflow")?;
            continue;
        }
        return len
            .checked_add(usize::from(byte))
            .ok_or_else(|| "Xiph length overflow".to_owned());
    }
}

/// Packs the three Vorbis headers into a Matroska `CodecPrivate` blob using
/// RFC 3533 / Matroska Xiph lacing (one byte of packet count minus one, two
/// laced lengths, then the three concatenated packets).
pub(crate) fn pack_xiph_laced_vorbis_headers(
    ident: &[u8],
    comment: &[u8],
    setup: &[u8],
) -> Result<Vec<u8>, String> {
    if !is_vorbis_header(ident, 0x01) || !is_vorbis_header(comment, 0x03) {
        return Err("invalid Vorbis ident or comment header".to_owned());
    }
    if !is_vorbis_header(setup, 0x05) {
        return Err("invalid Vorbis setup header".to_owned());
    }
    vorbis_ident_info(ident)?;
    let body_len = ident
        .len()
        .checked_add(comment.len())
        .and_then(|sum| sum.checked_add(setup.len()))
        .ok_or("Vorbis header size overflow")?;
    // One terminating byte per laced length plus the leading packet count.
    let lacing_len = (ident.len() / 255)
        .checked_add(1)
        .and_then(|n| n.checked_add(comment.len() / 255))
        .and_then(|n| n.checked_add(1))
        .ok_or("Vorbis header size overflow")?;
    let total = 1_usize
        .checked_add(lacing_len)
        .and_then(|n| n.checked_add(body_len))
        .ok_or("Vorbis header size overflow")?;
    if total > MAX_XIPH_VORBIS_HEADERS {
        return Err(format!(
            "Vorbis headers exceed {MAX_XIPH_VORBIS_HEADERS} bytes"
        ));
    }
    let mut out = Vec::with_capacity(total);
    out.push(0x02);
    push_xiph_length(&mut out, ident.len())?;
    push_xiph_length(&mut out, comment.len())?;
    out.extend_from_slice(ident);
    out.extend_from_slice(comment);
    out.extend_from_slice(setup);
    Ok(out)
}

/// Splits a Matroska `CodecPrivate` blob into its three Xiph-laced Vorbis
/// packets. Every length and offset is checked against the buffer and the
/// header signatures and identification fields are validated.
pub(crate) fn unpack_xiph_laced_vorbis_headers(data: &[u8]) -> Result<VorbisHeaders<'_>, String> {
    if data.len() > MAX_XIPH_VORBIS_HEADERS {
        return Err(format!(
            "Vorbis CodecPrivate exceeds {MAX_XIPH_VORBIS_HEADERS} bytes"
        ));
    }
    if data.first() != Some(&0x02) {
        return Err("Vorbis CodecPrivate must declare three packets".to_owned());
    }
    let mut idx = 1_usize;
    let ident_len = read_xiph_length(data, &mut idx)?;
    let comment_len = read_xiph_length(data, &mut idx)?;
    let ident_end = idx
        .checked_add(ident_len)
        .ok_or("Vorbis ident length overflow")?;
    let comment_end = ident_end
        .checked_add(comment_len)
        .ok_or("Vorbis comment length overflow")?;
    if comment_end >= data.len() {
        return Err("Vorbis setup packet is empty".to_owned());
    }
    let ident = data
        .get(idx..ident_end)
        .ok_or("truncated Vorbis identification header")?;
    let comment = data
        .get(ident_end..comment_end)
        .ok_or("truncated Vorbis comment header")?;
    let setup = data
        .get(comment_end..)
        .ok_or("truncated Vorbis setup header")?;
    if !is_vorbis_header(ident, 0x01) || !is_vorbis_header(comment, 0x03) {
        return Err("invalid Vorbis ident or comment header".to_owned());
    }
    if !is_vorbis_header(setup, 0x05) {
        return Err("invalid Vorbis setup header".to_owned());
    }
    vorbis_ident_info(ident)?;
    Ok((ident, comment, setup))
}

// ---- MP4 sample entry builders ------------------------------------------

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 6);
    out.push(tag);
    // MPEG-4 descriptor sizes are big-endian base-128 with a continuation bit
    // on every byte but the last.
    let mut parts = Vec::with_capacity(4);
    let mut remaining = body.len();
    loop {
        parts.push(u8::try_from(remaining & 0x7F).unwrap_or(0));
        remaining >>= 7;
        if remaining == 0 {
            break;
        }
    }
    for (index, part) in parts.iter().rev().enumerate() {
        let continuation = u8::from(index + 1 < parts.len()) << 7;
        out.push(part | continuation);
    }
    out.extend_from_slice(body);
    out
}

fn build_esds(object_type: u8, asc: &[u8]) -> Vec<u8> {
    let mut dsi_body = Vec::with_capacity(asc.len());
    dsi_body.extend_from_slice(asc);
    let dsi = descriptor(0x05, &dsi_body);
    let mut dcd_body = Vec::with_capacity(13 + dsi.len());
    dcd_body.push(object_type);
    dcd_body.push(0x15);
    dcd_body.extend_from_slice(&[0, 0, 0]);
    dcd_body.extend_from_slice(&[0; 4]);
    dcd_body.extend_from_slice(&[0; 4]);
    dcd_body.extend_from_slice(&dsi);
    let dcd = descriptor(0x04, &dcd_body);
    let mut es_body = Vec::with_capacity(3 + dcd.len());
    es_body.extend_from_slice(&[0, 1, 0]);
    es_body.extend_from_slice(&dcd);
    let sl = descriptor(0x06, &[2]);
    es_body.extend_from_slice(&sl);
    let es = descriptor(0x03, &es_body);
    let mut out = Vec::with_capacity(4 + es.len());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&es);
    out
}

pub(crate) fn build_aac_esds(asc: &[u8]) -> Vec<u8> {
    build_esds(0x40, asc)
}

fn full_box(kind: [u8; 4], body: &[u8]) -> Vec<u8> {
    let size = u32::try_from(body.len().saturating_add(8)).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(body);
    out
}

pub(crate) fn build_mp3_mp4a_sample_entry(channels: u16, sample_rate: u32) -> Vec<u8> {
    let mut entry = Vec::with_capacity(8 + 28 + 64);
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(b"mp4a");
    entry.extend_from_slice(&[0; 6]);
    entry.extend_from_slice(&1_u16.to_be_bytes());
    entry.extend_from_slice(&[0; 8]);
    entry.extend_from_slice(&channels.to_be_bytes());
    entry.extend_from_slice(&16_u16.to_be_bytes());
    entry.extend_from_slice(&[0; 4]);
    let rate = if sample_rate <= 0xFFFF {
        sample_rate << 16
    } else {
        0
    };
    entry.extend_from_slice(&rate.to_be_bytes());
    entry.extend_from_slice(&full_box(*b"esds", &build_esds(0x6B, &[])));
    let size = u32::try_from(entry.len()).unwrap_or(u32::MAX);
    entry[..4].copy_from_slice(&size.to_be_bytes());
    entry
}

pub(crate) fn build_vp09_sample_entry(width: u16, height: u16, vpcc_body: &[u8]) -> Vec<u8> {
    let mut entry = Vec::with_capacity(8 + 78 + vpcc_body.len() + 8);
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(b"vp09");
    entry.extend_from_slice(&[0; 6]);
    entry.extend_from_slice(&1_u16.to_be_bytes());
    entry.extend_from_slice(&[0; 16]);
    entry.extend_from_slice(&width.to_be_bytes());
    entry.extend_from_slice(&height.to_be_bytes());
    entry.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    entry.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(&1_u16.to_be_bytes());
    entry.extend_from_slice(&[0; 32]);
    entry.extend_from_slice(&0x0018_u16.to_be_bytes());
    entry.extend_from_slice(&0xFFFF_u16.to_be_bytes());
    entry.extend_from_slice(&full_box(*b"vpcC", vpcc_body));
    let size = u32::try_from(entry.len()).unwrap_or(u32::MAX);
    entry[..4].copy_from_slice(&size.to_be_bytes());
    entry
}

/// `QuickTime` `jpeg` visual sample entry with a 1:1 `pasp` box.
///
/// MJPEG samples are complete JPEG frames, so the entry has no codec-config
/// child box: the standard 86-byte `VisualSampleEntry` plus a 16-byte `pasp`
/// is the entire description. The 72 dpi resolution and `Photo - JPEG`
/// compressor name are the `QuickTime` conventions for JPEG video.
pub(crate) fn build_jpeg_sample_entry(width: u16, height: u16) -> Vec<u8> {
    let mut entry = Vec::with_capacity(8 + 78 + 16);
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(b"jpeg");
    entry.extend_from_slice(&[0; 6]); // reserved
    entry.extend_from_slice(&1_u16.to_be_bytes()); // data_reference_index
    entry.extend_from_slice(&[0; 2]); // pre_defined
    entry.extend_from_slice(&[0; 2]); // reserved
    entry.extend_from_slice(&[0; 12]); // pre_defined[3]
    entry.extend_from_slice(&width.to_be_bytes());
    entry.extend_from_slice(&height.to_be_bytes());
    entry.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // 72 dpi horizontal
    entry.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // 72 dpi vertical
    entry.extend_from_slice(&[0; 4]); // reserved
    entry.extend_from_slice(&1_u16.to_be_bytes()); // frame_count
    let mut compressor = [0_u8; 32];
    compressor[0] = 12;
    compressor[1..13].copy_from_slice(b"Photo - JPEG");
    entry.extend_from_slice(&compressor);
    entry.extend_from_slice(&0x0018_u16.to_be_bytes()); // depth: 24-bit
    entry.extend_from_slice(&0xFFFF_u16.to_be_bytes()); // color table ID
    entry.extend_from_slice(&16_u32.to_be_bytes());
    entry.extend_from_slice(b"pasp");
    entry.extend_from_slice(&1_u32.to_be_bytes());
    entry.extend_from_slice(&1_u32.to_be_bytes());
    let size = u32::try_from(entry.len()).unwrap_or(u32::MAX);
    entry[..4].copy_from_slice(&size.to_be_bytes());
    entry
}

/// Exact PCM sample representation carried by a container. Endianness and
/// depth are parsed from the source sample entry, never guessed at decode time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PcmFormat {
    U8,
    S16Le,
    S16Be,
    S24Le,
    S24Be,
    S32Le,
    S32Be,
    F32Le,
    F32Be,
    F64Le,
}

impl PcmFormat {
    pub(crate) const fn bytes_per_sample(self) -> u16 {
        match self {
            Self::U8 => 1,
            Self::S16Le | Self::S16Be => 2,
            Self::S24Le | Self::S24Be => 3,
            Self::S32Le | Self::S32Be | Self::F32Le | Self::F32Be => 4,
            Self::F64Le => 8,
        }
    }

    pub(crate) const fn is_big_endian(self) -> bool {
        matches!(self, Self::S16Be | Self::S24Be | Self::S32Be | Self::F32Be)
    }

    pub(crate) const fn is_float(self) -> bool {
        matches!(self, Self::F32Le | Self::F32Be | Self::F64Le)
    }

    pub(crate) const fn bit_depth(self) -> u16 {
        self.bytes_per_sample().saturating_mul(8)
    }
}

/// Upper bound on a FLAC configuration blob held in memory during remuxing.
const FLAC_CONFIG_CAP: usize = 1_048_576;

/// Validates a raw 34-byte FLAC STREAMINFO and returns
/// `(channels, sample_rate, bits_per_sample)`.
pub(crate) fn parse_flac_streaminfo(data: &[u8]) -> Result<(u16, u32, u8), String> {
    let si: [u8; 34] = data
        .try_into()
        .map_err(|_| "FLAC STREAMINFO must be exactly 34 bytes".to_owned())?;
    let min_block = u16::from_be_bytes([si[0], si[1]]);
    let max_block = u16::from_be_bytes([si[2], si[3]]);
    if min_block < 16 {
        return Err("FLAC STREAMINFO min_block_size is below 16".to_owned());
    }
    if max_block < min_block {
        return Err("FLAC STREAMINFO max_block_size is below min_block_size".to_owned());
    }
    let packed = u64::from_be_bytes([
        si[10], si[11], si[12], si[13], si[14], si[15], si[16], si[17],
    ]);
    let rate = u32::try_from((packed >> 44) & 0xF_FFFF)
        .map_err(|_| "FLAC sample rate exceeds 32 bits".to_owned())?;
    let channels = u16::try_from(((packed >> 41) & 0x7).saturating_add(1))
        .map_err(|_| "FLAC channel count overflow".to_owned())?;
    let bits = u8::try_from(((packed >> 36) & 0x1F).saturating_add(1))
        .map_err(|_| "FLAC bit depth overflow".to_owned())?;
    if !(1..=655_350).contains(&rate) {
        return Err(format!("FLAC STREAMINFO rate {rate} is out of range"));
    }
    if !(1..=8).contains(&channels) {
        return Err(format!(
            "FLAC STREAMINFO channel count {channels} is out of range"
        ));
    }
    if !(4..=32).contains(&bits) {
        return Err(format!("FLAC STREAMINFO bit depth {bits} is out of range"));
    }
    Ok((channels, rate, bits))
}

/// Walks a bounded FLAC metadata block chain and returns its STREAMINFO body.
/// `data` starts at the first metadata block header.
fn flac_metadata_chain(data: &[u8]) -> Result<[u8; 34], String> {
    if data.len() > FLAC_CONFIG_CAP {
        return Err(format!("FLAC metadata exceeds {FLAC_CONFIG_CAP} bytes"));
    }
    let mut idx = 0_usize;
    let mut streaminfo = None;
    loop {
        let header = *data
            .get(idx)
            .ok_or("truncated FLAC metadata block header")?;
        idx = idx.checked_add(1).ok_or("FLAC metadata offset overflow")?;
        let last = header & 0x80 != 0;
        let kind = header & 0x7F;
        let len_bytes = data
            .get(idx..idx.saturating_add(3))
            .ok_or("truncated FLAC metadata block length")?;
        let len = usize::try_from(u32::from_be_bytes([
            0,
            len_bytes[0],
            len_bytes[1],
            len_bytes[2],
        ]))
        .map_err(|_| "FLAC metadata block length overflow")?;
        idx = idx.checked_add(3).ok_or("FLAC metadata offset overflow")?;
        let end = idx.checked_add(len).ok_or("FLAC metadata block overflow")?;
        let body = data.get(idx..end).ok_or("truncated FLAC metadata block")?;
        idx = end;
        if streaminfo.is_none() {
            if kind != 0 || len != 34 {
                return Err("first FLAC metadata block must be a 34-byte STREAMINFO".to_owned());
            }
            let si: [u8; 34] = body
                .try_into()
                .map_err(|_| "invalid FLAC STREAMINFO length".to_owned())?;
            parse_flac_streaminfo(&si)?;
            streaminfo = Some(si);
        }
        if last {
            break;
        }
    }
    streaminfo.ok_or_else(|| "FLAC metadata has no STREAMINFO".to_owned())
}

/// Extracts the canonical 34-byte STREAMINFO from a Matroska `CodecPrivate`.
pub(crate) fn extract_flac_streaminfo_from_mkv_codec_private(
    data: &[u8],
) -> Result<[u8; 34], String> {
    if !data.starts_with(b"fLaC") {
        return Err("A_FLAC CodecPrivate must start with 'fLaC'".to_owned());
    }
    flac_metadata_chain(&data[4..])
}

/// Finds a child ISOBMFF box body inside a raw sample entry (or its `wave`).
pub(crate) fn find_child_box(raw: &[u8], start: usize, fourcc: [u8; 4]) -> Option<&[u8]> {
    let mut idx = start;
    while idx.checked_add(8)? <= raw.len() {
        let size = u32::from_be_bytes(raw.get(idx..idx.checked_add(4)?)?.try_into().ok()?);
        let kind = raw.get(idx + 4..idx + 8)?;
        // ISOBMFF size 1 means the real 64-bit size follows the 8-byte header;
        // using the marker itself as the size skips every largesize child.
        let (declared_size, header) = if size == 1 {
            let large = u64::from_be_bytes(<[u8; 8]>::try_from(raw.get(idx + 8..idx + 16)?).ok()?);
            (large, 16_usize)
        } else {
            (u64::from(size), 8_usize)
        };
        let body_start = idx.checked_add(header)?;
        let body_len = if size == 0 {
            raw.len().saturating_sub(body_start)
        } else {
            usize::try_from(declared_size).ok()?.checked_sub(header)?
        };
        let body_end = body_start.checked_add(body_len)?;
        if body_end > raw.len() {
            return None;
        }
        if kind == fourcc {
            return Some(&raw[body_start..body_end]);
        }
        if size == 0 {
            return None;
        }
        idx = body_end;
    }
    None
}

/// Extracts the `dfLa` box body from a raw FLAC audio sample entry.
pub(crate) fn extract_dfla_from_sample_entry(raw: &[u8]) -> Result<Vec<u8>, String> {
    let version = u16::from_be_bytes(
        raw.get(16..18)
            .ok_or("truncated FLAC sample entry")?
            .try_into()
            .map_err(|_| "invalid FLAC sample entry version".to_owned())?,
    );
    let start = 8 + match version {
        0 => 28,
        1 => 44,
        2 => 64,
        other => return Err(format!("unsupported FLAC sample entry version {other}")),
    };
    if let Some(dfla) = find_child_box(raw, start, *b"dfLa") {
        return Ok(dfla.to_vec());
    }
    let wave = find_child_box(raw, start, *b"wave").ok_or("FLAC sample entry is missing dfLa")?;
    find_child_box(wave, 0, *b"dfLa")
        .map(<[u8]>::to_vec)
        .ok_or_else(|| "FLAC sample entry is missing dfLa".to_owned())
}

/// Extracts the canonical STREAMINFO from a `dfLa` full box body.
/// Production muxing keeps the whole metadata chain; this helper is only used
/// by the container-helper tests that check the standalone `dfLa` parsing.
#[cfg(test)]
pub(crate) fn extract_flac_streaminfo_from_dfla(data: &[u8]) -> Result<[u8; 34], String> {
    if data.len() < 4 || data[..4] != [0, 0, 0, 0] {
        return Err("dfLa must be version 0 with zero flags".to_owned());
    }
    flac_metadata_chain(&data[4..])
}

/// Builds the canonical Matroska `A_FLAC` `CodecPrivate` from STREAMINFO.
pub(crate) fn build_mkv_flac_codec_private(streaminfo: &[u8; 34]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 4 + 34);
    out.extend_from_slice(b"fLaC");
    // Last metadata block, type STREAMINFO (0), 34-byte payload.
    out.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
    out.extend_from_slice(streaminfo);
    out
}

/// Builds an ISOBMFF `fLaC` audio sample entry with a compliant `dfLa` box.
pub(crate) fn build_flac_mp4_sample_entry(
    channels: u16,
    sample_rate: u32,
    streaminfo: &[u8; 34],
) -> Vec<u8> {
    let mut entry = Vec::with_capacity(8 + 28 + 8 + 42);
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(b"fLaC");
    entry.extend_from_slice(&[0; 6]);
    entry.extend_from_slice(&1_u16.to_be_bytes());
    entry.extend_from_slice(&[0; 8]);
    entry.extend_from_slice(&channels.to_be_bytes());
    // The sample entry's sample_size must agree with STREAMINFO's bit depth or
    // strict readers (e.g. Chromium) reject the track. Callers pass validated
    // STREAMINFO, so the legacy 16-bit default only covers an impossible parse
    // error and keeps 16-bit entries byte-for-byte unchanged.
    let sample_size =
        parse_flac_streaminfo(streaminfo).map_or(16_u16, |(_, _, bits)| u16::from(bits));
    entry.extend_from_slice(&sample_size.to_be_bytes());
    entry.extend_from_slice(&[0; 4]);
    let rate = if sample_rate <= 0xFFFF {
        sample_rate << 16
    } else {
        0
    };
    entry.extend_from_slice(&rate.to_be_bytes());
    let mut dfla = Vec::with_capacity(4 + 4 + 34);
    dfla.extend_from_slice(&[0, 0, 0, 0]);
    dfla.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
    dfla.extend_from_slice(streaminfo);
    entry.extend_from_slice(&full_box(*b"dfLa", &dfla));
    let size = u32::try_from(entry.len()).unwrap_or(u32::MAX);
    entry[..4].copy_from_slice(&size.to_be_bytes());
    entry
}

/// Builds a v0 `sowt` or v2 `lpcm` `QuickTime` PCM audio sample entry.
pub(crate) fn build_pcm_mp4_sample_entry(
    format: PcmFormat,
    channels: u16,
    sample_rate: u32,
) -> Vec<u8> {
    let v2 = sample_rate > 65_535 || format != PcmFormat::S16Le;
    if !v2 {
        let mut entry = Vec::with_capacity(8 + 28);
        entry.extend_from_slice(&[0; 4]);
        entry.extend_from_slice(b"sowt");
        entry.extend_from_slice(&[0; 6]);
        entry.extend_from_slice(&1_u16.to_be_bytes());
        entry.extend_from_slice(&[0; 8]);
        entry.extend_from_slice(&channels.to_be_bytes());
        entry.extend_from_slice(&16_u16.to_be_bytes());
        entry.extend_from_slice(&[0; 4]);
        entry.extend_from_slice(&(sample_rate << 16).to_be_bytes());
        let size = u32::try_from(entry.len()).unwrap_or(u32::MAX);
        entry[..4].copy_from_slice(&size.to_be_bytes());
        return entry;
    }
    let frame_bytes = u32::from(channels).saturating_mul(u32::from(format.bytes_per_sample()));
    let mut entry = Vec::with_capacity(8 + 64);
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(b"lpcm");
    entry.extend_from_slice(&[0; 6]);
    entry.extend_from_slice(&1_u16.to_be_bytes());
    entry.extend_from_slice(&2_u16.to_be_bytes()); // version
    entry.extend_from_slice(&[0; 6]); // revision and vendor
    // QuickTime Sound Sample Description v2 fixed placeholder fields.
    entry.extend_from_slice(&3_u16.to_be_bytes()); // always3
    entry.extend_from_slice(&16_u16.to_be_bytes()); // always16
    entry.extend_from_slice(&0xFFFE_u16.to_be_bytes()); // alwaysMinus2
    entry.extend_from_slice(&0_u16.to_be_bytes()); // always0
    entry.extend_from_slice(&0x0001_0000_u32.to_be_bytes()); // always65536
    entry.extend_from_slice(&72_u32.to_be_bytes()); // sizeOfStructOnly
    entry.extend_from_slice(&f64::from(sample_rate).to_be_bytes());
    entry.extend_from_slice(&u32::from(channels).to_be_bytes());
    entry.extend_from_slice(&0x7F00_0000_u32.to_be_bytes());
    entry.extend_from_slice(&u32::from(format.bit_depth()).to_be_bytes());
    // CoreAudio/QuickTime LPCM formatSpecificFlags:
    // bit 0 = isFloat, bit 1 = isBigEndian, bit 2 = isSignedInteger, bit 3 = isPacked.
    let mut flags = 0x08_u32;
    if format.is_float() {
        flags |= 0x01;
    } else if format != PcmFormat::U8 {
        flags |= 0x04;
    }
    if format.is_big_endian() {
        flags |= 0x02;
    }
    entry.extend_from_slice(&flags.to_be_bytes());
    entry.extend_from_slice(&frame_bytes.to_be_bytes()); // constBytesPerAudioPacket
    entry.extend_from_slice(&1_u32.to_be_bytes()); // constLPCMFramesPerAudioPacket
    let size = u32::try_from(entry.len()).unwrap_or(u32::MAX);
    entry[..4].copy_from_slice(&size.to_be_bytes());
    entry
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264_annex_b() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0, 1, 0x67, 0x42, 0x00, 0x1E]);
        out.extend_from_slice(&[0, 0, 1, 0x68, 0xCE]);
        out.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88]);
        out
    }

    #[test]
    fn h264_sps_vui_signals_color_matrix_and_range() {
        let mut bits = BitWriter::new();
        bits.write(0x42, 8); // profile_idc
        bits.write(0, 8); // constraint flags + reserved
        bits.write(30, 8); // level_idc
        bits.ue(0); // seq_parameter_set_id (baseline: no high-profile fields)
        bits.ue(4); // log2_max_frame_num_minus4
        bits.ue(0); // pic_order_cnt_type
        bits.ue(4); // log2_max_pic_order_cnt_lsb_minus4
        bits.ue(1); // max_num_ref_frames
        bits.write(0, 1); // gaps_in_frame_num_value_allowed_flag
        bits.ue(0); // pic_width_in_mbs_minus1
        bits.ue(0); // pic_height_in_map_units_minus1
        bits.write(1, 1); // frame_mbs_only_flag
        bits.write(1, 1); // direct_8x8_inference_flag
        bits.write(0, 1); // frame_cropping_flag
        bits.write(1, 1); // vui_parameters_present_flag
        bits.write(0, 1); // aspect_ratio_info_present_flag
        bits.write(0, 1); // overscan_info_present_flag
        bits.write(1, 1); // video_signal_type_present_flag
        bits.write(2, 3); // video_format
        bits.write(0, 1); // video_full_range_flag (limited)
        bits.write(1, 1); // colour_description_present_flag
        bits.write(1, 8); // colour_primaries
        bits.write(1, 8); // transfer_characteristics
        bits.write(1, 8); // matrix_coefficients = BT.709
        bits.write(1, 1); // rbsp_stop_one_bit
        let mut annex_b = vec![0, 0, 0, 1, 0x67];
        annex_b.extend(bits.bytes);
        annex_b.extend([0, 0, 0, 1, 0x68, 0xCE]);
        assert_eq!(
            h264_color_from_annex_b(&annex_b),
            Some(BitstreamColor {
                matrix: Some(1),
                full_range: false,
            })
        );
        assert_eq!(h264_dimensions_from_annex_b(&annex_b), Some((16, 16)));
        // A stream without a colour description must not claim one.
        assert_eq!(h264_color_from_annex_b(&h264_annex_b()), None);

        // Full range with colour_description_present_flag == 0 must keep full_range.
        let mut full_bits = BitWriter::new();
        full_bits.write(0x42, 8);
        full_bits.write(0, 8);
        full_bits.write(30, 8);
        full_bits.ue(0);
        full_bits.ue(4);
        full_bits.ue(0);
        full_bits.ue(4);
        full_bits.ue(1);
        full_bits.write(0, 1);
        full_bits.ue(0);
        full_bits.ue(0);
        full_bits.write(1, 1);
        full_bits.write(1, 1);
        full_bits.write(0, 1);
        full_bits.write(1, 1); // vui_parameters_present_flag
        full_bits.write(0, 1); // aspect_ratio_info_present_flag
        full_bits.write(0, 1); // overscan_info_present_flag
        full_bits.write(1, 1); // video_signal_type_present_flag
        full_bits.write(5, 3); // video_format = unspecified
        full_bits.write(1, 1); // video_full_range_flag = full
        full_bits.write(0, 1); // colour_description_present_flag = absent
        full_bits.write(1, 1); // rbsp_stop_one_bit
        let mut full_annex_b = vec![0, 0, 0, 1, 0x67];
        full_annex_b.extend(full_bits.bytes);
        assert_eq!(
            h264_color_from_annex_b(&full_annex_b),
            Some(BitstreamColor {
                matrix: None,
                full_range: true,
            })
        );
    }

    #[test]
    fn h264_annex_b_to_avcc_and_back() -> Result<(), String> {
        let annex_b = h264_annex_b();
        let avcc = annex_b_to_avcc(&annex_b);
        assert_eq!(
            avcc,
            vec![
                0, 0, 0, 4, 0x67, 0x42, 0x00, 0x1E, 0, 0, 0, 2, 0x68, 0xCE, 0, 0, 0, 2, 0x65, 0x88
            ]
        );
        let config = extract_avcc_from_annex_b(&annex_b).ok_or("avcC from annex B")?;
        assert_eq!(
            config,
            vec![
                1, 0x42, 0x00, 0x1E, 0xFF, 0xE1, 0, 4, 0x67, 0x42, 0x00, 0x1E, 1, 0, 2, 0x68, 0xCE
            ]
        );
        let out = avcc_to_annex_b(&config, &avcc, false)?;
        assert_eq!(
            out,
            vec![
                0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 0, 1, 0x68, 0xCE, 0, 0, 0, 1, 0x65, 0x88
            ]
        );
        assert!(is_h264_keyframe_annex_b(&annex_b));
        assert!(!is_h264_keyframe_annex_b(&[0, 0, 1, 0x41, 0x00]));
        Ok(())
    }

    #[test]
    fn h264_prepends_missing_param_sets() -> Result<(), String> {
        let config = extract_avcc_from_annex_b(&h264_annex_b()).ok_or("avcC")?;
        let (length_size, params) = avcc_param_sets(&config)?;
        assert_eq!(length_size, 4);
        assert_eq!(params.len(), 2);
        let slice = [0, 0, 0, 2, 0x65, 0x88];
        let out = avcc_to_annex_b(&config, &slice, true)?;
        assert_eq!(
            out,
            vec![
                0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E, 0, 0, 0, 1, 0x68, 0xCE, 0, 0, 0, 1, 0x65, 0x88
            ]
        );
        assert!(avcc_to_annex_b(&[1, 0], &slice, false).is_err());
        Ok(())
    }

    fn hevc_annex_b() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0, 1, 0x40, 0x01, 0x0C, 0x01]);
        out.extend_from_slice(&[0, 0, 1, 0x42, 0x01, 0x01, 0x01]);
        out.extend_from_slice(&[0, 0, 1, 0x44, 0x01, 0xC0]);
        out.extend_from_slice(&[0, 0, 1, 0x26, 0x01, 0x02]);
        out
    }

    #[test]
    fn hevc_annex_b_to_hvcc_and_back() -> Result<(), String> {
        let annex_b = hevc_annex_b();
        assert!(is_hevc_keyframe_annex_b(&annex_b));
        let config = extract_hvcc_from_annex_b(&annex_b).ok_or("hvcC")?;
        assert_eq!(config.len(), 49);
        assert_eq!(config[21] & 3, 3);
        assert_eq!(config[22], 3);
        let hvcc_sample = annex_b_to_hvcc(&annex_b);
        let out = hvcc_to_annex_b(&config, &hvcc_sample, false)?;
        assert!(out.ends_with(&[0, 0, 0, 1, 0x26, 0x01, 0x02]));
        assert!(extract_hvcc_from_annex_b(&[0, 0, 1, 0x26, 0x01]).is_none());
        Ok(())
    }

    #[test]
    fn av1_temporal_delimiter_and_config() -> Result<(), String> {
        let obu = [0x0A, 0x02, 0x00, 0x0A];
        let with_td = ensure_av1_temporal_delimiter(&obu);
        assert_eq!(with_td[..2], [0x12, 0x00]);
        assert_eq!(&with_td[2..], obu.as_slice());
        assert_eq!(ensure_av1_temporal_delimiter(&with_td), with_td);
        let config = extract_av1c_from_obu(&obu).ok_or("av1C")?;
        assert_eq!(config[0], 0x81);
        assert_eq!(config[1] & 0x1F, 0);
        assert_eq!(config[1] >> 5, 0);
        assert_eq!(&config[4..], obu.as_slice());
        assert!(!is_av1_keyframe(&obu));
        Ok(())
    }

    #[test]
    fn av1_keyframes_inspect_frame_header_bits() {
        let key_frame = [0x1A, 0x01, 0x00]; // OBU_FRAME_HEADER, frame_type KEY_FRAME
        let inter_frame = [0x1A, 0x01, 0x40]; // frame_type INTER_FRAME
        let shown_frame = [0x1A, 0x01, 0x80]; // show_existing_frame = 1
        assert!(is_av1_keyframe(&key_frame));
        assert!(!is_av1_keyframe(&inter_frame));
        assert!(!is_av1_keyframe(&shown_frame));
        assert!(!is_av1_keyframe(&[]));
    }

    #[test]
    fn av1c_parses_sequence_header_bitfields() -> Result<(), String> {
        // profile=2, level=10 (>7, so the tier bit is present), tier=1.
        let seq_obu = [0x0A, 0x04, 0x40, 0x00, 0x00, 0x54];
        let config = extract_av1c_from_obu(&seq_obu).ok_or("av1C")?;
        assert_eq!(config[0], 0x81);
        assert_eq!(config[1], 0x4A);
        assert_eq!(config[2], 0x8C);
        assert_eq!(config[3], 0x00);
        assert_eq!(&config[4..], seq_obu.as_slice());
        Ok(())
    }

    #[test]
    fn aac_asc_adts_round_trip() -> Result<(), String> {
        let asc = [0x12, 0x10];
        assert_eq!(parse_audio_specific_config(&asc), Some((2, 4, 2)));
        let frame = wrap_adts(&asc, &[0xAA; 8])?;
        assert_eq!(frame.len(), 15);
        assert_eq!(&frame[..2], &[0xFF, 0xF1]);
        let frames = unwrap_adts(&frame);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, vec![0xAA; 8]);
        assert_eq!(frames[0].1, asc);
        assert!(wrap_adts(&[0x00], &[]).is_err());
        assert_eq!(unwrap_adts(b"not adts").len(), 0);
        Ok(())
    }

    #[test]
    fn opus_heads_and_ts_access_units() -> Result<(), String> {
        // dOps: version 0 and big-endian fields.
        let dops = [0_u8, 2, 0x01, 0x38, 0x00, 0x00, 0xBB, 0x80, 0x00, 0x00, 0];
        assert_eq!(dops[0], 0);
        assert_eq!(&dops[4..8], &[0x00, 0x00, 0xBB, 0x80]);
        let head = dops_to_opus_head(&dops)?;
        assert_eq!(&head[..8], b"OpusHead");
        assert_eq!(head[8], 1);
        assert_eq!(head[9], 2);
        assert_eq!(&head[12..16], &[0x80, 0xBB, 0x00, 0x00]);
        assert_eq!(head.len(), 19);
        assert!(dops_to_opus_head(&[0; 4]).is_err());
        let mut bad_version = dops;
        bad_version[0] = 1;
        assert!(dops_to_opus_head(&bad_version).is_err());
        let round_trip = opus_head_to_dops(&head)?;
        assert_eq!(round_trip[0], 0);
        assert_eq!(round_trip, dops);
        let default_head = default_opus_head(2, 312);
        assert_eq!(default_head.len(), 19);
        assert_eq!(&default_head[10..12], &312_u16.to_le_bytes());
        let packets = [vec![1_u8; 300], vec![2_u8; 5]];
        let mut stream = Vec::new();
        for packet in &packets {
            stream.extend_from_slice(&wrap_ts_opus(packet));
        }
        assert_eq!(unwrap_ts_opus(&stream), packets);
        Ok(())
    }

    #[test]
    fn opus_packet_duration_follows_rfc_6716_toc() {
        // CELT 2.5 ms (config 16), one frame.
        assert_eq!(opus_packet_samples_48k(&[0x80]), Some(120));
        // CELT 10 ms (config 30): 0xF0 and 0xF4 are both one frame, while
        // frame-count code 1 (0xF5) carries two.
        assert_eq!(opus_packet_samples_48k(&[0xF0]), Some(480));
        assert_eq!(opus_packet_samples_48k(&[0xF4]), Some(480));
        assert_eq!(opus_packet_samples_48k(&[0xF5]), Some(960));
        // CELT 20 ms (config 31), one frame (0xFC).
        assert_eq!(opus_packet_samples_48k(&[0xFC]), Some(960));
        // SILK 40 ms (config 2): one frame (0x14) and two via code 1 (0x15).
        assert_eq!(opus_packet_samples_48k(&[0x14]), Some(1920));
        assert_eq!(opus_packet_samples_48k(&[0x15]), Some(3840));
        // Hybrid 10 ms (config 12) and 20 ms (config 13).
        assert_eq!(opus_packet_samples_48k(&[0x60]), Some(480));
        assert_eq!(opus_packet_samples_48k(&[0x68]), Some(960));
        // Code 3 reads an explicit frame count from the second TOC byte.
        assert_eq!(opus_packet_samples_48k(&[0xF3, 0x03]), Some(1440));
        assert_eq!(opus_packet_samples_48k(&[0xF3, 0x00]), None);
        // Malformed or over-long packets are rejected.
        assert_eq!(opus_packet_samples_48k(&[]), None);
        assert_eq!(opus_packet_samples_48k(&[0xF3]), None);
        assert_eq!(opus_packet_samples_48k(&[0xFF, 0x0C]), None);
    }

    #[test]
    fn mp4_sample_entries_are_well_formed() -> Result<(), String> {
        let esds = build_aac_esds(&[0x12, 0x10]);
        assert_eq!(&esds[..4], &[0, 0, 0, 0]);
        assert_eq!(esds[4], 0x03);
        let mp3 = build_mp3_mp4a_sample_entry(2, 44_100);
        assert_eq!(&mp3[4..8], b"mp4a");
        let mp3_size = u32::from_be_bytes(
            mp3.get(..4)
                .ok_or("mp4a size")?
                .try_into()
                .map_err(|_| "mp4a size bytes")?,
        );
        assert_eq!(usize::try_from(mp3_size).unwrap_or(0), mp3.len());
        assert_eq!(&mp3[24..26], &2_u16.to_be_bytes());
        let vp9 = build_vp09_sample_entry(1280, 720, &[1, 0, 0, 0, 2, 30, 0xA1, 1, 1, 1, 0, 0]);
        assert_eq!(&vp9[4..8], b"vp09");
        assert_eq!(&vp9[32..36], &[0x05, 0x00, 0x02, 0xD0]);
        assert_eq!(&vp9[36..40], &[0x00, 0x48, 0x00, 0x00]);
        let vp9_size = u32::from_be_bytes(
            vp9.get(..4)
                .ok_or("vp09 size")?
                .try_into()
                .map_err(|_| "vp09 size bytes")?,
        );
        assert_eq!(usize::try_from(vp9_size).unwrap_or(0), vp9.len());
        Ok(())
    }

    #[test]
    fn jpeg_sample_entry_is_well_formed() -> Result<(), String> {
        let entry = build_jpeg_sample_entry(320, 240);
        assert_eq!(&entry[4..8], b"jpeg");
        // Standard 86-byte VisualSampleEntry plus a 16-byte pasp box.
        assert_eq!(entry.len(), 102);
        assert_eq!(&entry[8..14], &[0; 6]);
        assert_eq!(&entry[14..16], &1_u16.to_be_bytes());
        assert_eq!(&entry[32..34], &320_u16.to_be_bytes());
        assert_eq!(&entry[34..36], &240_u16.to_be_bytes());
        assert_eq!(&entry[36..40], &0x0048_0000_u32.to_be_bytes());
        assert_eq!(&entry[40..44], &0x0048_0000_u32.to_be_bytes());
        assert_eq!(&entry[48..50], &1_u16.to_be_bytes());
        assert_eq!(entry[50], 12);
        assert_eq!(&entry[51..63], b"Photo - JPEG");
        assert_eq!(&entry[82..84], &0x0018_u16.to_be_bytes());
        assert_eq!(&entry[84..86], &0xFFFF_u16.to_be_bytes());
        assert_eq!(&entry[86..90], &16_u32.to_be_bytes());
        assert_eq!(&entry[90..94], b"pasp");
        assert_eq!(&entry[94..98], &1_u32.to_be_bytes());
        assert_eq!(&entry[98..102], &1_u32.to_be_bytes());
        let size = u32::from_be_bytes(
            entry
                .get(..4)
                .ok_or("jpeg size")?
                .try_into()
                .map_err(|_| "jpeg size bytes")?,
        );
        assert_eq!(usize::try_from(size).unwrap_or(0), entry.len());
        Ok(())
    }

    #[test]
    fn avcc_keeps_all_distinct_parameter_sets() -> Result<(), String> {
        let mut annex_b = Vec::new();
        for nalu in [
            &[0x67, 0x42, 0x00, 0x1E][..],
            &[0x68, 0xCE][..],
            &[0x67, 0x42, 0x00, 0x1E][..], // duplicate SPS
            &[0x67, 0x64, 0x00, 0x1F][..],
            &[0x68, 0xCB][..],
        ] {
            annex_b.push(0);
            annex_b.push(0);
            annex_b.push(1);
            annex_b.extend_from_slice(nalu);
        }
        let config = extract_avcc_from_annex_b(&annex_b).ok_or("avcC")?;
        let (_, sets) = avcc_param_sets(&config)?;
        assert_eq!(config[5], 0xE2);
        assert_eq!(config[18], 2);
        assert_eq!(sets.len(), 4);
        assert_eq!(sets[0], [0x67, 0x42, 0x00, 0x1E]);
        assert_eq!(sets[1], [0x67, 0x64, 0x00, 0x1F]);
        assert_eq!(sets[2], [0x68, 0xCE]);
        assert_eq!(sets[3], [0x68, 0xCB]);
        Ok(())
    }

    #[test]
    fn hvcc_header_uses_emulation_prevention_free_profile_tier_level() -> Result<(), String> {
        // SPS: header, one prefix byte, then a 12-byte profile_tier_level whose
        // compatibility flags contain an emulation-prevention byte.
        let mut annex_b = vec![
            0, 0, 1, 0x42, 0x01, 0x01, 0x42, 0x00, 0x00, 0x03, 0xFF, 0xAA, 0x11, 0x22, 0x33, 0x44,
            0x55, 0x66, 0x5A,
        ];
        annex_b.extend_from_slice(&[0, 0, 1, 0x40, 0x01, 0x01, 0x01]);
        annex_b.extend_from_slice(&[0, 0, 1, 0x44, 0x01, 0xC0]);
        let config = extract_hvcc_from_annex_b(&annex_b).ok_or("hvcC")?;
        // config[0] is configurationVersion. The SPS prefix byte is skipped;
        // profile and level are checked independently at their own offsets.
        assert_eq!(config[1], 0x42, "general_profile_space/tier/idc");
        assert_eq!(&config[2..6], &[0x00, 0x00, 0xFF, 0xAA]);
        assert_eq!(config[12], 0x5A, "general_level_idc");
        assert_eq!(
            &config[13..23],
            &[0xF0, 0x00, 0xFC, 0xFC, 0xF8, 0xF8, 0x00, 0x00, 0x03, 3]
        );
        // The NALU arrays still carry the raw, unmodified SPS bytes.
        assert!(config.windows(16).any(|w| w
            == [
                0x42, 0x01, 0x01, 0x42, 0x00, 0x00, 0x03, 0xFF, 0xAA, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x5A
            ]));
        Ok(())
    }

    #[test]
    fn esds_descriptor_lengths_use_base128_encoding() {
        let esds = build_aac_esds(&[0x12; 200]);
        // 200 = 0x81 0x48 in base-128 with continuation bits.
        assert!(esds.windows(3).any(|w| w == [0x05, 0x81, 0x48]));
        let small = build_aac_esds(&[0x12, 0x10]);
        assert!(small.windows(3).any(|w| w == [0x05, 0x02, 0x12]));
    }

    struct BitWriter {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit: 0,
            }
        }

        fn write(&mut self, value: u32, count: usize) {
            for shift in (0..count).rev() {
                let index = self.bit / 8;
                if index == self.bytes.len() {
                    self.bytes.push(0);
                }
                if shift < 32 && (value >> shift.min(31)) & 1 == 1 {
                    self.bytes[index] |= 1 << (7 - (self.bit % 8));
                }
                self.bit += 1;
            }
        }

        fn ue(&mut self, value: u32) {
            let code = value.saturating_add(1);
            let width = 32_u32.saturating_sub(code.leading_zeros()) as usize;
            self.write(0, width.saturating_sub(1));
            self.write(code, width);
        }

        fn se(&mut self, value: i32) {
            let code = if value <= 0 {
                value.checked_mul(-2).and_then(|v| u32::try_from(v).ok())
            } else {
                value
                    .checked_mul(2)
                    .and_then(|v| v.checked_sub(1))
                    .and_then(|v| u32::try_from(v).ok())
            };
            self.ue(code.unwrap_or(0));
        }
    }

    #[test]
    fn h264_sps_yields_real_dimensions() {
        let mut bits = BitWriter::new();
        bits.write(66, 8); // profile_idc
        bits.write(0, 8);
        bits.write(30, 8);
        bits.ue(0); // seq_parameter_set_id
        bits.ue(0); // log2_max_frame_num_minus4
        bits.ue(0); // pic_order_cnt_type
        bits.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        bits.ue(1); // max_num_ref_frames
        bits.write(0, 1); // gaps_in_frame_num_value_allowed_flag
        bits.ue(79); // pic_width_in_mbs_minus1 -> 1280
        bits.ue(44); // pic_height_in_map_units_minus1 -> 720
        bits.write(1, 1); // frame_mbs_only_flag
        bits.write(0, 1); // direct_8x8_inference_flag
        bits.write(0, 1); // frame_cropping_flag
        bits.write(0, 1); // vui_parameters_present_flag
        let mut annex_b = vec![0, 0, 0, 1, 0x67];
        annex_b.extend_from_slice(&bits.bytes);
        assert_eq!(h264_dimensions_from_annex_b(&annex_b), Some((1280, 720)));
    }

    #[test]
    fn hevc_sps_yields_real_dimensions_and_cropping() {
        let mut bits = BitWriter::new();
        bits.write(0, 4); // sps_video_parameter_set_id
        bits.write(0, 3); // sps_max_sub_layers_minus1
        bits.write(1, 1); // temporal id nesting
        bits.write(0x01, 8); // profile_space/tier/profile_idc
        bits.write(0xFFFF_FFFF, 32); // compatibility flags
        bits.write(0, 48); // constraint flags
        bits.write(120, 8); // level_idc
        bits.ue(0); // sps_seq_parameter_set_id
        bits.ue(1); // chroma_format_idc = 4:2:0
        bits.ue(1920);
        bits.ue(1088);
        bits.write(1, 1); // conformance_window_flag
        bits.ue(0);
        bits.ue(0);
        bits.ue(0);
        bits.ue(2); // bottom crop: 2 * 2 = 4 luma lines -> 1084
        let mut annex_b = vec![0, 0, 0, 1, 0x42, 0x01];
        annex_b.extend_from_slice(&bits.bytes);
        assert_eq!(hevc_dimensions_from_annex_b(&annex_b), Some((1920, 1084)));
    }

    #[test]
    fn h264_scaling_list_stops_when_next_scale_reaches_zero() {
        let mut bits = BitWriter::new();
        bits.write(100, 8); // high profile
        bits.write(0, 8);
        bits.write(30, 8);
        bits.ue(0); // sps_id
        bits.ue(1); // chroma_format_idc = 4:2:0
        bits.ue(0); // bit_depth_luma_minus8
        bits.ue(0); // bit_depth_chroma_minus8
        bits.write(0, 1); // qpprime_y_zero_transform_bypass_flag
        bits.write(1, 1); // seq_scaling_matrix_present_flag
        bits.write(1, 1); // seq_scaling_list_present_flag[0]
        bits.se(-8); // first delta_scale: nextScale becomes 0
        // The remaining 15 entries repeat lastScale and carry no bits.
        for _ in 1..8 {
            bits.write(0, 1); // other scaling lists absent
        }
        bits.ue(0); // log2_max_frame_num_minus4
        bits.ue(0); // pic_order_cnt_type
        bits.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        bits.ue(1); // max_num_ref_frames
        bits.write(0, 1); // gaps_in_frame_num_value_allowed_flag
        bits.ue(79); // 1280
        bits.ue(44); // 720
        bits.write(1, 1); // frame_mbs_only_flag
        bits.write(0, 1); // direct_8x8_inference_flag
        bits.write(0, 1); // frame_cropping_flag
        bits.write(0, 1); // vui_parameters_present_flag
        let mut annex_b = vec![0, 0, 0, 1, 0x67];
        annex_b.extend_from_slice(&bits.bytes);
        assert_eq!(h264_dimensions_from_annex_b(&annex_b), Some((1280, 720)));
    }

    #[test]
    fn h264_10bit_crop_uses_chroma_units_only() {
        let mut bits = BitWriter::new();
        bits.write(100, 8); // high 10 profile
        bits.write(0, 8);
        bits.write(30, 8);
        bits.ue(0); // sps_id
        bits.ue(1); // chroma_format_idc = 4:2:0
        bits.ue(2); // bit_depth_luma_minus8 = 10-bit
        bits.ue(2); // bit_depth_chroma_minus8 = 10-bit
        bits.write(0, 1); // transform bypass
        bits.write(0, 1); // no scaling matrix
        bits.ue(0);
        bits.ue(0);
        bits.ue(0);
        bits.ue(1);
        bits.write(0, 1);
        bits.ue(119); // 1920
        bits.ue(67); // 1088 coded lines
        bits.write(1, 1); // frame_mbs_only_flag
        bits.write(0, 1);
        bits.write(1, 1); // frame_cropping_flag
        bits.ue(0); // left
        bits.ue(0); // right
        bits.ue(0); // top
        bits.ue(4); // bottom: 4 * SubHeightC(2) = 8 lines -> 1080
        bits.write(0, 1); // vui
        let mut annex_b = vec![0, 0, 0, 1, 0x67];
        annex_b.extend_from_slice(&bits.bytes);
        assert_eq!(h264_dimensions_from_annex_b(&annex_b), Some((1920, 1080)));
    }

    fn hevc_sps_with_sublayers(
        sub0: (u32, u32),
        sub1: (u32, u32),
        sub0_payload: fn(&mut BitWriter),
        sub1_payload: fn(&mut BitWriter),
    ) -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.write(0, 4); // sps_video_parameter_set_id
        bits.write(2, 3); // sps_max_sub_layers_minus1 = 2
        bits.write(1, 1); // temporal id nesting
        bits.write(1, 8); // general profile
        bits.write(0, 32); // compatibility flags
        bits.write(0, 48); // constraint flags
        bits.write(120, 8); // general level
        // All presence flags, then reserved_zero_2bits for sublayers 2..8,
        // then the payloads in sublayer order (H.265 §7.3.3).
        bits.write(sub0.0, 1);
        bits.write(sub0.1, 1);
        bits.write(sub1.0, 1);
        bits.write(sub1.1, 1);
        bits.write(0, 12);
        sub0_payload(&mut bits);
        sub1_payload(&mut bits);
        bits.ue(0); // sps_id
        bits.ue(1); // chroma_format_idc = 4:2:0
        bits.ue(1920);
        bits.ue(1080);
        bits.write(0, 1); // no conformance window
        let mut annex_b = vec![0, 0, 0, 1, 0x42, 0x01];
        annex_b.extend_from_slice(&bits.bytes);
        annex_b
    }

    #[test]
    fn hevc_sps_reads_all_sublayer_flags_before_payloads() {
        let profile_payload = |bits: &mut BitWriter| {
            bits.write(2, 8); // sub_layer_profile_idc
            bits.write(0, 32); // compatibility flags
            bits.write(0, 32); // constraint flags high
            bits.write(0, 16); // constraint flags low: 88 bits in total
        };
        let level_payload = |bits: &mut BitWriter| bits.write(123, 8);

        // Sublayer 0 has a profile payload, sublayer 1 a level payload.
        let annex_b = hevc_sps_with_sublayers((1, 0), (0, 1), profile_payload, level_payload);
        assert_eq!(hevc_dimensions_from_annex_b(&annex_b), Some((1920, 1080)));

        // Sublayer 0 has a level payload and sublayer 1 a profile payload.
        // Splitting payloads into all-profiles-then-all-levels loops (or
        // reading reserved bits after them) misparses this stream.
        let annex_b = hevc_sps_with_sublayers((0, 1), (1, 0), level_payload, profile_payload);
        assert_eq!(hevc_dimensions_from_annex_b(&annex_b), Some((1920, 1080)));
    }

    #[test]
    fn vp9_keyframe_yields_color_config() {
        // marker=2, profile=2, show_existing=0, key frame, show_frame=1,
        // sync code, then ten_or_twelve_bit=0, CS_UNSPECIFIED, full_range=0.
        let frame = [0x92, 0x49, 0x83, 0x42, 0x00];
        assert_eq!(
            vp9_color_from_keyframe(&frame),
            Some(Vp9KeyframeColor {
                profile: 2,
                bit_depth: 10,
                chroma_subsampling: 0,
                full_range: false,
                colour_primaries: 2,
                transfer: 2,
                matrix: 2,
            })
        );
        let mut twelve_bit = frame;
        twelve_bit[4] = 0x80; // ten_or_twelve_bit and CS_UNSPECIFIED
        assert_eq!(
            vp9_color_from_keyframe(&twelve_bit),
            Some(Vp9KeyframeColor {
                bit_depth: 12,
                ..Vp9KeyframeColor {
                    profile: 2,
                    bit_depth: 10,
                    chroma_subsampling: 0,
                    full_range: false,
                    colour_primaries: 2,
                    transfer: 2,
                    matrix: 2,
                }
            })
        );
        assert_eq!(
            vp9_color_from_keyframe(&[0x82, 0x49, 0x83, 0x42, 0x00]),
            Some(Vp9KeyframeColor {
                profile: 0,
                bit_depth: 8,
                chroma_subsampling: 0,
                full_range: false,
                colour_primaries: 2,
                transfer: 2,
                matrix: 2,
            })
        );
        // Inter frames, show-existing frames and bad sync codes establish nothing.
        assert_eq!(
            vp9_color_from_keyframe(&[0x96, 0x49, 0x83, 0x42, 0x00]),
            None
        );
        assert_eq!(
            vp9_color_from_keyframe(&[0x9A, 0x49, 0x83, 0x42, 0x00]),
            None
        );
        assert_eq!(
            vp9_color_from_keyframe(&[0x92, 0x00, 0x00, 0x00, 0x00]),
            None
        );
        assert_eq!(vp9_color_from_keyframe(&[0x92]), None);
    }

    #[test]
    fn vp9_keyframe_color_config_rgb_and_422() {
        // Profile 1, CS_RGB, reserved_zero=0: 4:4:4 full-range sRGB.
        assert_eq!(
            vp9_color_from_keyframe(&[0xA2, 0x49, 0x83, 0x42, 0xE0]),
            Some(Vp9KeyframeColor {
                profile: 1,
                bit_depth: 8,
                chroma_subsampling: 3,
                full_range: true,
                colour_primaries: 1,
                transfer: 13,
                matrix: 0,
            })
        );
        // Profile 1, BT.709 with subsampling_x=1, subsampling_y=0: 4:2:2.
        assert_eq!(
            vp9_color_from_keyframe(&[0xA2, 0x49, 0x83, 0x42, 0x48]),
            Some(Vp9KeyframeColor {
                profile: 1,
                bit_depth: 8,
                chroma_subsampling: 2,
                full_range: false,
                colour_primaries: 1,
                transfer: 1,
                matrix: 1,
            })
        );
        // Profile 2, 12-bit BT.2020 has transfer characteristic 15.
        assert_eq!(
            vp9_color_from_keyframe(&[0x92, 0x49, 0x83, 0x42, 0xD0]),
            Some(Vp9KeyframeColor {
                profile: 2,
                bit_depth: 12,
                chroma_subsampling: 0,
                full_range: false,
                colour_primaries: 9,
                transfer: 15,
                matrix: 9,
            })
        );
    }

    #[test]
    fn vp9_keyframe_rejects_invalid_color_config() {
        // CS_RGB is only legal for profiles 1 and 3.
        assert_eq!(
            vp9_color_from_keyframe(&[0x82, 0x49, 0x83, 0x42, 0xE0]),
            None
        );
        // (0, 1) subsampling is not one of the allowed chroma layouts.
        assert_eq!(
            vp9_color_from_keyframe(&[0xA2, 0x49, 0x83, 0x42, 0x44]),
            None
        );
        // color_space 6 is reserved.
        assert_eq!(
            vp9_color_from_keyframe(&[0xA2, 0x49, 0x83, 0x42, 0xC0]),
            None
        );
        // Truncated color_config must not read zero-padding as metadata.
        assert_eq!(vp9_color_from_keyframe(&[0xA2, 0x49, 0x83, 0x42]), None);
    }

    fn vorbis_ident(blocksize: u8) -> Vec<u8> {
        let mut ident = vec![0_u8; 30];
        ident[0] = 0x01;
        ident[1..7].copy_from_slice(b"vorbis");
        ident[7..11].copy_from_slice(&0_u32.to_le_bytes());
        ident[11] = 2;
        ident[12..16].copy_from_slice(&44_100_u32.to_le_bytes());
        ident[28] = blocksize;
        ident[29] = 1;
        ident
    }

    fn vorbis_comment(extra: usize) -> Vec<u8> {
        let mut comment = vec![0_u8; 7usize.saturating_add(extra)];
        comment[0] = 0x03;
        comment[1..7].copy_from_slice(b"vorbis");
        comment
    }

    fn vorbis_setup(extra: usize) -> Vec<u8> {
        let mut setup = vec![0_u8; 7usize.saturating_add(extra)];
        setup[0] = 0x05;
        setup[1..7].copy_from_slice(b"vorbis");
        setup
    }

    #[test]
    fn vorbis_headers_pack_and_unpack_roundtrip() -> Result<(), String> {
        let ident = vorbis_ident(0xB8);
        let comment = vorbis_comment(255);
        let setup = vorbis_setup(600);
        let packed = pack_xiph_laced_vorbis_headers(&ident, &comment, &setup)?;
        assert_eq!(packed[0], 0x02);
        let (out_ident, out_comment, out_setup) = unpack_xiph_laced_vorbis_headers(&packed)?;
        assert_eq!(out_ident, ident);
        assert_eq!(out_comment, comment);
        assert_eq!(out_setup, setup);
        assert_eq!(vorbis_ident_info(out_ident)?, (2, 44_100, 8, 11));
        Ok(())
    }

    #[test]
    fn vorbis_lacing_handles_exact_255_byte_packets() -> Result<(), String> {
        let ident = vorbis_ident(0x88);
        let comment = vorbis_comment(248); // exactly 255 bytes total
        let setup = vorbis_setup(503); // exactly 510 bytes, a multiple of 255
        let packed = pack_xiph_laced_vorbis_headers(&ident, &comment, &setup)?;
        assert_eq!(packed[1], u8::try_from(ident.len()).unwrap_or(0));
        assert_eq!(&packed[2..4], &[0xFF, 0x00]);
        let (out_ident, out_comment, out_setup) = unpack_xiph_laced_vorbis_headers(&packed)?;
        assert_eq!(out_ident, ident);
        assert_eq!(out_comment.len(), 255);
        assert_eq!(out_setup.len(), 510);
        Ok(())
    }

    #[test]
    fn vorbis_pack_rejects_invalid_headers() {
        let ident = vorbis_ident(0xB8);
        let comment = vorbis_comment(0);
        let setup = vorbis_setup(0);
        assert!(pack_xiph_laced_vorbis_headers(&[], &comment, &setup).is_err());
        assert!(pack_xiph_laced_vorbis_headers(&ident, &[], &setup).is_err());
        assert!(pack_xiph_laced_vorbis_headers(&ident, &comment, &[]).is_err());
        let mut bad = ident.clone();
        bad[0] = 0x02;
        assert!(pack_xiph_laced_vorbis_headers(&bad, &comment, &setup).is_err());
        let mut bad = comment.clone();
        bad[0] = 0x01;
        assert!(pack_xiph_laced_vorbis_headers(&ident, &bad, &setup).is_err());
        let mut bad = setup.clone();
        bad[0] = 0x03;
        assert!(pack_xiph_laced_vorbis_headers(&ident, &comment, &bad).is_err());
        let huge = vec![0_u8; MAX_XIPH_VORBIS_HEADERS];
        assert!(pack_xiph_laced_vorbis_headers(&ident, &comment, &huge).is_err());
    }

    #[test]
    fn vorbis_unpack_rejects_malformed_codec_private() -> Result<(), String> {
        let ident = vorbis_ident(0xB8);
        let comment = vorbis_comment(0);
        let setup = vorbis_setup(0);
        let good = pack_xiph_laced_vorbis_headers(&ident, &comment, &setup)?;

        assert!(unpack_xiph_laced_vorbis_headers(&[]).is_err());
        assert!(unpack_xiph_laced_vorbis_headers(&[0x01, 30]).is_err());
        // Truncated lacing that ends on 0xFF.
        assert!(unpack_xiph_laced_vorbis_headers(&[0x02, 0xFF]).is_err());
        // Declared length exceeds the buffer.
        assert!(unpack_xiph_laced_vorbis_headers(&[0x02, 0xFF, 0x02, 0x03]).is_err());
        // Empty setup packet: the laced lengths consume the whole buffer.
        let mut no_setup = vec![0x02];
        push_xiph_length(&mut no_setup, ident.len())?;
        push_xiph_length(&mut no_setup, comment.len())?;
        no_setup.extend_from_slice(&ident);
        no_setup.extend_from_slice(&comment);
        assert!(unpack_xiph_laced_vorbis_headers(&no_setup).is_err());

        // Each of the three packet signatures must be intact.
        let ident_start = 3;
        let comment_start = ident_start + ident.len();
        let setup_start = comment_start + comment.len();
        for start in [ident_start, comment_start, setup_start] {
            let mut bad = good.clone();
            if let Some(byte) = bad.get_mut(start) {
                *byte ^= 0x01;
                assert!(
                    unpack_xiph_laced_vorbis_headers(&bad).is_err(),
                    "mutated packet at byte {start} was accepted"
                );
            }
        }

        // A bad ident field inside an otherwise well-formed blob is rejected.
        let mut bad_ident = ident.clone();
        bad_ident[11] = 0;
        let mut blob = vec![0x02];
        push_xiph_length(&mut blob, bad_ident.len())?;
        push_xiph_length(&mut blob, comment.len())?;
        blob.extend_from_slice(&bad_ident);
        blob.extend_from_slice(&comment);
        blob.extend_from_slice(&setup);
        assert!(unpack_xiph_laced_vorbis_headers(&blob).is_err());

        let oversized = vec![0_u8; MAX_XIPH_VORBIS_HEADERS + 1];
        assert!(unpack_xiph_laced_vorbis_headers(&oversized).is_err());
        Ok(())
    }

    #[test]
    fn vorbis_ident_info_checks_version_channels_rate_blocks_and_framing() {
        let ident = vorbis_ident(0xB8);
        assert_eq!(vorbis_ident_info(&ident), Ok((2, 44_100, 8, 11)));
        assert!(vorbis_ident_info(&[]).is_err());
        assert!(vorbis_ident_info(&ident[..29]).is_err());
        let mut bad = ident.clone();
        bad[7] = 1;
        assert!(vorbis_ident_info(&bad).is_err());
        let mut bad = ident.clone();
        bad[11] = 0;
        assert!(vorbis_ident_info(&bad).is_err());
        let mut bad = ident.clone();
        bad[12..16].copy_from_slice(&0_u32.to_le_bytes());
        assert!(vorbis_ident_info(&bad).is_err());
        let mut bad = ident.clone();
        bad[28] = 0x05; // bs0=5 below the 6..=13 range
        assert!(vorbis_ident_info(&bad).is_err());
        let mut bad = ident.clone();
        bad[28] = 0xBD; // bs0=13 > bs1=11
        assert!(vorbis_ident_info(&bad).is_err());
        let mut bad = ident.clone();
        bad[29] = 0;
        assert!(vorbis_ident_info(&bad).is_err());
    }

    fn flac_streaminfo(rate: u32, channels: u16, bits: u8) -> [u8; 34] {
        let mut si = [0_u8; 34];
        si[0..2].copy_from_slice(&4096_u16.to_be_bytes());
        si[2..4].copy_from_slice(&4096_u16.to_be_bytes());
        let packed =
            (u64::from(rate) << 44) | (u64::from(channels - 1) << 41) | (u64::from(bits - 1) << 36);
        si[10..18].copy_from_slice(&packed.to_be_bytes());
        si
    }

    #[test]
    fn flac_streaminfo_roundtrips_through_container_helpers() -> Result<(), String> {
        let si = flac_streaminfo(48_000, 2, 16);
        assert_eq!(parse_flac_streaminfo(&si)?, (2, 48_000, 16));
        let mkv = build_mkv_flac_codec_private(&si);
        assert_eq!(&mkv[..4], b"fLaC");
        assert_eq!(&mkv[4..8], &[0x80, 0x00, 0x00, 0x22]);
        assert_eq!(extract_flac_streaminfo_from_mkv_codec_private(&mkv)?, si);

        let entry = build_flac_mp4_sample_entry(2, 48_000, &si);
        assert_eq!(&entry[4..8], b"fLaC");
        let dfla = extract_dfla_from_sample_entry(&entry)?;
        assert_eq!(dfla[..4], [0, 0, 0, 0]);
        assert_eq!(extract_flac_streaminfo_from_dfla(&dfla)?, si);
        Ok(())
    }

    #[test]
    fn flac_mp4_sample_entry_sample_size_follows_streaminfo() -> Result<(), String> {
        let si16 = flac_streaminfo(48_000, 2, 16);
        let si24 = flac_streaminfo(48_000, 2, 24);
        let entry16 = build_flac_mp4_sample_entry(2, 48_000, &si16);
        let entry24 = build_flac_mp4_sample_entry(2, 48_000, &si24);

        // The AudioSampleEntry sample_size at bytes 26..28 must match STREAMINFO.
        assert_eq!(&entry16[26..28], &16_u16.to_be_bytes());
        assert_eq!(&entry24[26..28], &24_u16.to_be_bytes());

        // The 16-bit entry stays byte-for-byte legacy: moving STREAMINFO to 24
        // bits changes only the declared size and the embedded STREAMINFO body.
        let mut legacy = entry24.clone();
        legacy[26..28].copy_from_slice(&16_u16.to_be_bytes());
        legacy[52..86].copy_from_slice(&si16);
        assert_eq!(entry16, legacy);

        let dfla16 = extract_dfla_from_sample_entry(&entry16)?;
        assert_eq!(extract_flac_streaminfo_from_dfla(&dfla16)?, si16);
        let dfla24 = extract_dfla_from_sample_entry(&entry24)?;
        assert_eq!(extract_flac_streaminfo_from_dfla(&dfla24)?, si24);
        Ok(())
    }

    #[test]
    fn find_child_box_reads_isobmff_largesize_children() {
        // A 64-bit-size `free` box must be skipped, not treated as a 1-byte box.
        let mut raw = Vec::new();
        raw.extend_from_slice(&1_u32.to_be_bytes());
        raw.extend_from_slice(b"free");
        raw.extend_from_slice(&20_u64.to_be_bytes());
        raw.extend_from_slice(&[0_u8; 4]);
        let dfla_body = [7_u8, 8, 9];
        raw.extend_from_slice(&1_u32.to_be_bytes());
        raw.extend_from_slice(b"dfLa");
        raw.extend_from_slice(&19_u64.to_be_bytes());
        raw.extend_from_slice(&dfla_body);
        assert_eq!(find_child_box(&raw, 0, *b"dfLa"), Some(&dfla_body[..]));

        let mut normal = Vec::new();
        normal.extend_from_slice(&11_u32.to_be_bytes());
        normal.extend_from_slice(b"dfLa");
        normal.extend_from_slice(&[1_u8, 2, 3]);
        assert_eq!(find_child_box(&normal, 0, *b"dfLa"), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn flac_helpers_reject_malformed_config() {
        let mut si = flac_streaminfo(48_000, 2, 16);
        si[0..2].copy_from_slice(&8_u16.to_be_bytes());
        assert!(parse_flac_streaminfo(&si).is_err());
        si[0..2].copy_from_slice(&4096_u16.to_be_bytes());
        si[2..4].copy_from_slice(&16_u16.to_be_bytes());
        assert!(parse_flac_streaminfo(&si).is_err());
        let mut bad = build_mkv_flac_codec_private(&flac_streaminfo(48_000, 2, 16));
        bad[4] = 0x01;
        assert!(extract_flac_streaminfo_from_mkv_codec_private(&bad).is_err());
        bad[4] = 0x80;
        bad[7] = 0x21;
        assert!(extract_flac_streaminfo_from_mkv_codec_private(&bad).is_err());
        assert!(extract_dfla_from_sample_entry(&[0; 8]).is_err());
        assert!(extract_flac_streaminfo_from_dfla(&[1, 0, 0, 0]).is_err());
    }

    #[test]
    fn pcm_format_sizes_are_exact() {
        assert_eq!(PcmFormat::U8.bytes_per_sample(), 1);
        assert_eq!(PcmFormat::S24Be.bytes_per_sample(), 3);
        assert_eq!(PcmFormat::F64Le.bytes_per_sample(), 8);
        assert!(PcmFormat::S16Be.is_big_endian());
        assert!(!PcmFormat::S16Le.is_big_endian());
        assert!(PcmFormat::F32Le.is_float());
        assert!(!PcmFormat::S32Le.is_float());
        let entry = build_pcm_mp4_sample_entry(PcmFormat::S16Le, 2, 44_100);
        assert_eq!(&entry[4..8], b"sowt");
        let entry = build_pcm_mp4_sample_entry(PcmFormat::F32Le, 2, 96_000);
        assert_eq!(&entry[4..8], b"lpcm");
        assert_eq!(entry.len(), 8 + 64);
    }
}
