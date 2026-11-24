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

pub(crate) fn annex_b_to_avcc(annex_b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(annex_b.len() + 16);
    for nalu in annex_b_nalus(annex_b) {
        if nalu_type(nalu) == Some(9) {
            continue;
        }
        let Ok(len) = u32::try_from(nalu.len()) else {
            continue;
        };
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }
    out
}

pub(crate) fn extract_avcc_from_annex_b(annex_b: &[u8]) -> Option<Vec<u8>> {
    let mut sps = None;
    let mut pps = None;
    for nalu in annex_b_nalus(annex_b) {
        match nalu_type(nalu) {
            Some(7) if sps.is_none() => sps = Some(nalu.to_vec()),
            Some(8) if pps.is_none() => pps = Some(nalu.to_vec()),
            _ => {}
        }
    }
    let sps = sps?;
    let pps = pps?;
    let profile = sps.get(1).copied()?;
    let compat = sps.get(2).copied()?;
    let level = sps.get(3).copied()?;
    let sps_len = u16::try_from(sps.len()).ok()?;
    let pps_len = u16::try_from(pps.len()).ok()?;
    let mut out = Vec::with_capacity(9 + sps.len() + pps.len());
    out.extend_from_slice(&[1, profile, compat, level, 0xFF, 0xE1]);
    out.extend_from_slice(&sps_len.to_be_bytes());
    out.extend_from_slice(&sps);
    out.push(1);
    out.extend_from_slice(&pps_len.to_be_bytes());
    out.extend_from_slice(&pps);
    Some(out)
}

pub(crate) fn is_h264_keyframe_annex_b(annex_b: &[u8]) -> bool {
    annex_b_nalus(annex_b)
        .iter()
        .any(|n| nalu_type(n) == Some(5))
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
    let mut out = Vec::with_capacity(annex_b.len() + 16);
    for nalu in annex_b_nalus(annex_b) {
        if hevc_nalu_type(nalu) == Some(35) {
            continue;
        }
        let Ok(len) = u32::try_from(nalu.len()) else {
            continue;
        };
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }
    out
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
    let profile_byte = first_sps.get(2).copied().unwrap_or(0x01);
    let level = first_sps.get(13).copied().unwrap_or(120);
    let mut out = Vec::with_capacity(23 + vps.len() * 8 + sps.len() * 8 + pps.len() * 8);
    out.push(1);
    out.push(profile_byte);
    out.extend_from_slice(first_sps.get(3..7).unwrap_or(&[0xFF; 4]));
    out.extend_from_slice(first_sps.get(7..13).unwrap_or(&[0; 6]));
    out.push(level);
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

pub(crate) fn is_hevc_keyframe_annex_b(annex_b: &[u8]) -> bool {
    annex_b_nalus(annex_b)
        .iter()
        .any(|n| matches!(hevc_nalu_type(n), Some(16..=21)))
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

/// MSB-first bit reader for AV1 sequence headers. Bits past the end read as
/// zero so a truncated header still yields a usable (if conservative) config.
struct Av1BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> Av1BitReader<'a> {
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
}

pub(crate) fn extract_av1c_from_obu(obu_data: &[u8]) -> Option<Vec<u8>> {
    let seq = obus(obu_data).into_iter().find(|obu| obu.kind == 1)?;
    let payload = seq.bytes.get(seq.payload_start..)?;
    if payload.is_empty() {
        return None;
    }
    let mut bits = Av1BitReader::new(payload);
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
    obus(obu_data)
        .iter()
        .any(|obu| obu.kind == 1 || obu.kind == 3)
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
    if aot == 0 || aot == 31 || freq_index > 15 || channels == 0 || channels > 15 {
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

// ---- MP4 sample entry builders ------------------------------------------

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 2);
    out.push(tag);
    let len = u8::try_from(body.len()).unwrap_or(u8::MAX);
    out.push(len);
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

pub(crate) fn build_vp09_sample_entry(
    width: u16,
    height: u16,
    vpcc_body: Option<&[u8]>,
) -> Vec<u8> {
    let default_body = [1_u8, 0, 0, 0, 0, 10, 0x80, 1, 1, 1, 0, 0];
    let body = vpcc_body.unwrap_or(&default_body);
    let mut entry = Vec::with_capacity(8 + 78 + body.len() + 8);
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
    entry.extend_from_slice(&full_box(*b"vpcC", body));
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
        assert!(is_av1_keyframe(&obu));
        Ok(())
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
        let vp9 = build_vp09_sample_entry(1280, 720, None);
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
}
