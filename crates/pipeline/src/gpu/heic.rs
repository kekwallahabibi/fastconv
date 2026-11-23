use std::fmt;

/// Color metadata written to the HEIF `colr` (`nclx`) box to match the HEVC VUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeicColorInfo {
    pub colour_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range: bool,
}

impl HeicColorInfo {
    /// BT.709 primaries (1), sRGB transfer (13), SMPTE 170M / BT.601 matrix (6), limited range.
    ///
    /// Matches NVENC's internal ABGR-to-YUV BT.601 limited-range color conversion.
    pub const BT601_LIMITED: Self = Self {
        colour_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
        full_range: false,
    };
}

/// Errors returned while parsing an HEVC Annex-B bitstream or assembling a HEIC container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeicMuxError {
    InvalidNal(&'static str),
    MissingNal(&'static str),
    InvalidSps(&'static str),
    SizeOverflow(&'static str),
}

impl fmt::Display for HeicMuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNal(reason) => write!(f, "invalid HEVC NAL unit: {reason}"),
            Self::MissingNal(kind) => write!(f, "HEVC bitstream is missing required {kind} NAL"),
            Self::InvalidSps(reason) => write!(f, "invalid HEVC SPS header: {reason}"),
            Self::SizeOverflow(field) => write!(f, "HEIC container size overflow in {field}"),
        }
    }
}

impl std::error::Error for HeicMuxError {}

const NAL_VPS: u8 = 32;
const NAL_SPS: u8 = 33;
const NAL_PPS: u8 = 34;
const NAL_AUD: u8 = 35;

/// Wraps a single HEVC intra frame (Annex-B bitstream containing VPS, SPS, PPS, and IDR NALs)
/// into an ISO/IEC 23008-12 HEIC (`heic`/`mif1`/`miaf`) still-image container.
///
/// # Errors
///
/// Returns [`HeicMuxError`] if any NAL header is truncated, required HEVC parameter sets or VCL
/// slice NALs are missing, the SPS header cannot be parsed for `hvcC`, or payload sizes exceed
/// ISO BMFF limits.
pub fn mux_annex_b_hevc_to_heic(
    annex_b: &[u8],
    width: u32,
    height: u32,
    color: HeicColorInfo,
) -> Result<Vec<u8>, HeicMuxError> {
    let mut vps_nals: Vec<&[u8]> = Vec::new();
    let mut sps_nals: Vec<&[u8]> = Vec::new();
    let mut pps_nals: Vec<&[u8]> = Vec::new();
    let mut picture_nals: Vec<&[u8]> = Vec::new();
    let mut has_vcl = false;

    for nal in split_annex_b_nals(annex_b) {
        let (Some(&hdr0), true) = (nal.first(), nal.len() >= 2) else {
            return Err(HeicMuxError::InvalidNal(
                "HEVC NAL shorter than 2-byte header",
            ));
        };
        let nal_type = (hdr0 >> 1) & 0x3F;
        match nal_type {
            NAL_VPS => vps_nals.push(nal),
            NAL_SPS => sps_nals.push(nal),
            NAL_PPS => pps_nals.push(nal),
            NAL_AUD => {}
            other => {
                if other <= 31 {
                    has_vcl = true;
                }
                picture_nals.push(nal);
            }
        }
    }

    if vps_nals.is_empty() {
        return Err(HeicMuxError::MissingNal("VPS"));
    }
    let Some(first_sps) = sps_nals.first().copied() else {
        return Err(HeicMuxError::MissingNal("SPS"));
    };
    if pps_nals.is_empty() {
        return Err(HeicMuxError::MissingNal("PPS"));
    }
    if !has_vcl {
        return Err(HeicMuxError::MissingNal("VCL/IDR"));
    }

    let sps_info = parse_sps_config(first_sps)?;
    let hvcc_payload = build_hvcc_payload(&sps_info, &vps_nals, &sps_nals, &pps_nals)?;

    let mut mdat_payload = Vec::with_capacity(annex_b.len());
    for nal in &picture_nals {
        let len_u32 =
            u32::try_from(nal.len()).map_err(|_| HeicMuxError::SizeOverflow("mdat NAL length"))?;
        mdat_payload.extend_from_slice(&len_u32.to_be_bytes());
        mdat_payload.extend_from_slice(nal);
    }
    let mdat_len_u32 = u32::try_from(mdat_payload.len())
        .map_err(|_| HeicMuxError::SizeOverflow("mdat payload"))?;

    let ftyp = build_ftyp_box()?;
    let hdlr = build_hdlr_box()?;
    let pitm = build_pitm_box()?;
    let iinf = build_iinf_box()?;
    let iprp = build_iprp_box(&hvcc_payload, width, height, color)?;
    let placeholder_iloc = build_iloc_box(0, mdat_len_u32)?;

    // `iloc` uses a fixed 4-byte `base_offset`, so `meta` size is invariant to the offset value.
    let meta_size = 12_usize
        .checked_add(hdlr.len())
        .and_then(|n| n.checked_add(pitm.len()))
        .and_then(|n| n.checked_add(placeholder_iloc.len()))
        .and_then(|n| n.checked_add(iinf.len()))
        .and_then(|n| n.checked_add(iprp.len()))
        .ok_or(HeicMuxError::SizeOverflow("meta box"))?;

    let mdat_payload_offset = ftyp
        .len()
        .checked_add(meta_size)
        .and_then(|n| n.checked_add(8))
        .ok_or(HeicMuxError::SizeOverflow("mdat offset"))?;
    let mdat_offset_u32 = u32::try_from(mdat_payload_offset)
        .map_err(|_| HeicMuxError::SizeOverflow("iloc offset"))?;
    let iloc = build_iloc_box(mdat_offset_u32, mdat_len_u32)?;

    let mut meta_children = Vec::with_capacity(meta_size);
    meta_children.extend_from_slice(&hdlr);
    meta_children.extend_from_slice(&pitm);
    meta_children.extend_from_slice(&iloc);
    meta_children.extend_from_slice(&iinf);
    meta_children.extend_from_slice(&iprp);
    let meta = wrap_full_box(*b"meta", 0, 0, &meta_children)?;
    let mdat = wrap_box(*b"mdat", &mdat_payload)?;

    let mut output = Vec::with_capacity(ftyp.len() + meta.len() + mdat.len());
    output.extend_from_slice(&ftyp);
    output.extend_from_slice(&meta);
    output.extend_from_slice(&mdat);
    Ok(output)
}

fn split_annex_b_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0_usize;
    while i + 2 < data.len() {
        if data.get(i..i + 4) == Some(&[0, 0, 0, 1]) {
            starts.push((i, i + 4));
            i += 4;
        } else if data.get(i..i + 3) == Some(&[0, 0, 1]) {
            starts.push((i, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }

    let mut nals = Vec::with_capacity(starts.len());
    for (idx, &(_, nal_start)) in starts.iter().enumerate() {
        let nal_end = starts
            .get(idx + 1)
            .map_or(data.len(), |&(next_code_start, _)| next_code_start);
        if let Some(slice) = data.get(nal_start..nal_end)
            && !slice.is_empty()
        {
            nals.push(slice);
        }
    }
    nals
}

struct SpsConfig {
    general_profile_space_tier_idc: u8,
    general_profile_compatibility_flags: [u8; 4],
    general_constraint_indicator_flags: [u8; 6],
    general_level_idc: u8,
    max_sub_layers_minus1: u8,
    temporal_id_nested: u8,
    chroma_format_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
}

fn parse_sps_config(sps_nal: &[u8]) -> Result<SpsConfig, HeicMuxError> {
    let payload = sps_nal.get(2..).ok_or(HeicMuxError::InvalidSps(
        "SPS shorter than 2-byte NAL header",
    ))?;
    let rbsp = unescape_rbsp(payload);
    let mut reader = BitReader::new(&rbsp);

    let _vps_id = reader.read_bits(4)?;
    let max_sub_layers_minus1 = u8::try_from(reader.read_bits(3)?)
        .map_err(|_| HeicMuxError::InvalidSps("max_sub_layers_minus1 overflow"))?;
    let temporal_id_nested = u8::try_from(reader.read_bits(1)?)
        .map_err(|_| HeicMuxError::InvalidSps("temporal_id_nested overflow"))?;

    let general_profile_space_tier_idc = u8::try_from(reader.read_bits(8)?)
        .map_err(|_| HeicMuxError::InvalidSps("profile byte overflow"))?;
    let mut general_profile_compatibility_flags = [0_u8; 4];
    for slot in &mut general_profile_compatibility_flags {
        *slot = u8::try_from(reader.read_bits(8)?)
            .map_err(|_| HeicMuxError::InvalidSps("compatibility flags overflow"))?;
    }
    let mut general_constraint_indicator_flags = [0_u8; 6];
    for slot in &mut general_constraint_indicator_flags {
        *slot = u8::try_from(reader.read_bits(8)?)
            .map_err(|_| HeicMuxError::InvalidSps("constraint flags overflow"))?;
    }
    let general_level_idc = u8::try_from(reader.read_bits(8)?)
        .map_err(|_| HeicMuxError::InvalidSps("level idc overflow"))?;

    if max_sub_layers_minus1 > 0 {
        let mut sub_layer_flags = [(false, false); 8];
        let Some(active_flags) = sub_layer_flags.get_mut(..usize::from(max_sub_layers_minus1))
        else {
            return Err(HeicMuxError::InvalidSps(
                "max_sub_layers_minus1 out of bounds",
            ));
        };
        for (profile_present, level_present) in active_flags.iter_mut() {
            *profile_present = reader.read_bits(1)? != 0;
            *level_present = reader.read_bits(1)? != 0;
        }
        for _ in max_sub_layers_minus1..8 {
            let _ = reader.read_bits(2)?;
        }
        for &(profile_present, level_present) in active_flags.iter() {
            if profile_present {
                reader.skip_bits(88)?;
            }
            if level_present {
                reader.skip_bits(8)?;
            }
        }
    }

    let _sps_id = reader.read_ue()?;
    let chroma_format_u32 = reader.read_ue()?;
    let chroma_format_idc = u8::try_from(chroma_format_u32)
        .map_err(|_| HeicMuxError::InvalidSps("chroma_format_idc"))?;
    if chroma_format_idc > 3 {
        return Err(HeicMuxError::InvalidSps("chroma_format_idc out of 0..=3"));
    }
    if chroma_format_idc == 3 {
        let _separate_colour_plane = reader.read_bits(1)?;
    }

    let _pic_width = reader.read_ue()?;
    let _pic_height = reader.read_ue()?;
    let conformance_window = reader.read_bits(1)?;
    if conformance_window != 0 {
        for _ in 0..4 {
            let _ = reader.read_ue()?;
        }
    }

    let bit_depth_luma_minus8 = u8::try_from(reader.read_ue()?)
        .map_err(|_| HeicMuxError::InvalidSps("bit_depth_luma_minus8"))?;
    let bit_depth_chroma_minus8 = u8::try_from(reader.read_ue()?)
        .map_err(|_| HeicMuxError::InvalidSps("bit_depth_chroma_minus8"))?;
    if bit_depth_luma_minus8 > 7 || bit_depth_chroma_minus8 > 7 {
        return Err(HeicMuxError::InvalidSps("bit depth out of 8..=15 range"));
    }

    Ok(SpsConfig {
        general_profile_space_tier_idc,
        general_profile_compatibility_flags,
        general_constraint_indicator_flags,
        general_level_idc,
        max_sub_layers_minus1,
        temporal_id_nested,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
    })
}

fn unescape_rbsp(nal_payload: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(nal_payload.len());
    let mut zeros = 0_usize;
    for &byte in nal_payload {
        if zeros == 2 && byte == 0x03 {
            zeros = 0;
            continue;
        }
        rbsp.push(byte);
        if byte == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    rbsp
}

struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read_bits(&mut self, count: u32) -> Result<u32, HeicMuxError> {
        let mut value = 0_u32;
        for _ in 0..count {
            let byte_idx = self.bit_pos / 8;
            let bit_in_byte = 7 - (self.bit_pos % 8);
            let byte = *self
                .data
                .get(byte_idx)
                .ok_or(HeicMuxError::InvalidSps("unexpected end of SPS RBSP"))?;
            value = (value << 1) | u32::from((byte >> bit_in_byte) & 1);
            self.bit_pos += 1;
        }
        Ok(value)
    }

    fn skip_bits(&mut self, count: usize) -> Result<(), HeicMuxError> {
        let total_bits = self.data.len().saturating_mul(8);
        let next = self
            .bit_pos
            .checked_add(count)
            .ok_or(HeicMuxError::InvalidSps("bit offset overflow"))?;
        if next > total_bits {
            return Err(HeicMuxError::InvalidSps("unexpected end of SPS RBSP"));
        }
        self.bit_pos = next;
        Ok(())
    }

    fn read_ue(&mut self) -> Result<u32, HeicMuxError> {
        let mut leading_zeros = 0_u32;
        while self.read_bits(1)? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(HeicMuxError::InvalidSps("Exp-Golomb code exceeds 31 bits"));
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(leading_zeros)?;
        let base = 1_u32
            .checked_shl(leading_zeros)
            .and_then(|v| v.checked_sub(1))
            .ok_or(HeicMuxError::InvalidSps("Exp-Golomb shift overflow"))?;
        base.checked_add(suffix)
            .ok_or(HeicMuxError::InvalidSps("Exp-Golomb value overflow"))
    }
}

fn build_hvcc_payload(
    sps: &SpsConfig,
    vps_nals: &[&[u8]],
    sps_nals: &[&[u8]],
    pps_nals: &[&[u8]],
) -> Result<Vec<u8>, HeicMuxError> {
    let mut out = Vec::with_capacity(128);
    // configurationVersion = 1 (ISO/IEC 14496-15 §8.3.3.1.2)
    out.push(1);
    out.push(sps.general_profile_space_tier_idc);
    out.extend_from_slice(&sps.general_profile_compatibility_flags);
    out.extend_from_slice(&sps.general_constraint_indicator_flags);
    out.push(sps.general_level_idc);
    // reserved(4)=1111 | min_spatial_segmentation_idc(12)=0
    out.extend_from_slice(&0xF000_u16.to_be_bytes());
    // reserved(6)=111111 | parallelismType(2)=0
    out.push(0xFC);
    // reserved(6)=111111 | chromaFormat(2)
    out.push(0xFC | (sps.chroma_format_idc & 0x03));
    // reserved(5)=11111 | bitDepthLumaMinus8(3)
    out.push(0xF8 | (sps.bit_depth_luma_minus8 & 0x07));
    // reserved(5)=11111 | bitDepthChromaMinus8(3)
    out.push(0xF8 | (sps.bit_depth_chroma_minus8 & 0x07));
    // avgFrameRate(16)=0
    out.extend_from_slice(&0_u16.to_be_bytes());
    // constantFrameRate(2)=0 | numTemporalLayers(3) | temporalIdNested(1) | lengthSizeMinusOne(2)=3
    let num_temporal_layers = (sps.max_sub_layers_minus1.saturating_add(1)) & 0x07;
    let packed_flags = (num_temporal_layers << 3) | ((sps.temporal_id_nested & 1) << 2) | 3;
    out.push(packed_flags);
    // numOfArrays = 3 (VPS, SPS, PPS)
    out.push(3);

    for (nal_type, nals) in [
        (NAL_VPS, vps_nals),
        (NAL_SPS, sps_nals),
        (NAL_PPS, pps_nals),
    ] {
        // array_completeness(1)=1 | reserved(1)=0 | NAL_unit_type(6)
        out.push(0x80 | (nal_type & 0x3F));
        let count_u16 =
            u16::try_from(nals.len()).map_err(|_| HeicMuxError::SizeOverflow("hvcC NAL count"))?;
        out.extend_from_slice(&count_u16.to_be_bytes());
        for nal in nals {
            let nal_len_u16 = u16::try_from(nal.len())
                .map_err(|_| HeicMuxError::SizeOverflow("hvcC parameter NAL length"))?;
            out.extend_from_slice(&nal_len_u16.to_be_bytes());
            out.extend_from_slice(nal);
        }
    }

    Ok(out)
}

fn build_ftyp_box() -> Result<Vec<u8>, HeicMuxError> {
    let mut payload = Vec::with_capacity(20);
    payload.extend_from_slice(b"heic");
    payload.extend_from_slice(&0_u32.to_be_bytes());
    payload.extend_from_slice(b"mif1");
    payload.extend_from_slice(b"heic");
    payload.extend_from_slice(b"miaf");
    wrap_box(*b"ftyp", &payload)
}

fn build_hdlr_box() -> Result<Vec<u8>, HeicMuxError> {
    let mut payload = Vec::with_capacity(21);
    payload.extend_from_slice(&0_u32.to_be_bytes());
    payload.extend_from_slice(b"pict");
    payload.extend_from_slice(&[0_u8; 12]);
    payload.push(0);
    wrap_full_box(*b"hdlr", 0, 0, &payload)
}

fn build_pitm_box() -> Result<Vec<u8>, HeicMuxError> {
    wrap_full_box(*b"pitm", 0, 0, &1_u16.to_be_bytes())
}

fn build_iloc_box(base_offset: u32, extent_length: u32) -> Result<Vec<u8>, HeicMuxError> {
    let mut payload = Vec::with_capacity(22);
    // offset_size=4, length_size=4
    payload.push(0x44);
    // base_offset_size=4, reserved=0
    payload.push(0x40);
    // item_count = 1
    payload.extend_from_slice(&1_u16.to_be_bytes());
    // item_ID = 1, data_reference_index = 0
    payload.extend_from_slice(&1_u16.to_be_bytes());
    payload.extend_from_slice(&0_u16.to_be_bytes());
    // base_offset = offset of mdat payload
    payload.extend_from_slice(&base_offset.to_be_bytes());
    // extent_count = 1, extent_offset = 0, extent_length
    payload.extend_from_slice(&1_u16.to_be_bytes());
    payload.extend_from_slice(&0_u32.to_be_bytes());
    payload.extend_from_slice(&extent_length.to_be_bytes());
    wrap_full_box(*b"iloc", 0, 0, &payload)
}

fn build_iinf_box() -> Result<Vec<u8>, HeicMuxError> {
    let mut infe_payload = Vec::with_capacity(9);
    infe_payload.extend_from_slice(&1_u16.to_be_bytes());
    infe_payload.extend_from_slice(&0_u16.to_be_bytes());
    infe_payload.extend_from_slice(b"hvc1");
    infe_payload.push(0);
    let infe = wrap_full_box(*b"infe", 2, 0, &infe_payload)?;

    let mut iinf_payload = Vec::with_capacity(2 + infe.len());
    iinf_payload.extend_from_slice(&1_u16.to_be_bytes());
    iinf_payload.extend_from_slice(&infe);
    wrap_full_box(*b"iinf", 0, 0, &iinf_payload)
}

fn build_iprp_box(
    hvcc_payload: &[u8],
    width: u32,
    height: u32,
    color: HeicColorInfo,
) -> Result<Vec<u8>, HeicMuxError> {
    let hvcc = wrap_box(*b"hvcC", hvcc_payload)?;

    let mut ispe_payload = Vec::with_capacity(8);
    ispe_payload.extend_from_slice(&width.to_be_bytes());
    ispe_payload.extend_from_slice(&height.to_be_bytes());
    let ispe = wrap_full_box(*b"ispe", 0, 0, &ispe_payload)?;

    let pixi = wrap_full_box(*b"pixi", 0, 0, &[3, 8, 8, 8])?;

    let mut colr_payload = Vec::with_capacity(11);
    colr_payload.extend_from_slice(b"nclx");
    colr_payload.extend_from_slice(&color.colour_primaries.to_be_bytes());
    colr_payload.extend_from_slice(&color.transfer_characteristics.to_be_bytes());
    colr_payload.extend_from_slice(&color.matrix_coefficients.to_be_bytes());
    colr_payload.push(if color.full_range { 0x80 } else { 0x00 });
    let color_box = wrap_box(*b"colr", &colr_payload)?;

    let mut ipco_payload =
        Vec::with_capacity(hvcc.len() + ispe.len() + pixi.len() + color_box.len());
    ipco_payload.extend_from_slice(&hvcc);
    ipco_payload.extend_from_slice(&ispe);
    ipco_payload.extend_from_slice(&pixi);
    ipco_payload.extend_from_slice(&color_box);
    let ipco = wrap_box(*b"ipco", &ipco_payload)?;

    // Property indices are 1-based in ipco order: 1=hvcC (essential), 2=ispe, 3=pixi, 4=colr.
    let mut ipma_payload = Vec::with_capacity(11);
    ipma_payload.extend_from_slice(&1_u32.to_be_bytes());
    ipma_payload.extend_from_slice(&1_u16.to_be_bytes());
    ipma_payload.push(4);
    ipma_payload.extend_from_slice(&[0x81, 0x02, 0x03, 0x04]);
    let ipma = wrap_full_box(*b"ipma", 0, 0, &ipma_payload)?;

    let mut iprp_payload = Vec::with_capacity(ipco.len() + ipma.len());
    iprp_payload.extend_from_slice(&ipco);
    iprp_payload.extend_from_slice(&ipma);
    wrap_box(*b"iprp", &iprp_payload)
}

fn wrap_box(fourcc: [u8; 4], payload: &[u8]) -> Result<Vec<u8>, HeicMuxError> {
    let total_len = payload
        .len()
        .checked_add(8)
        .ok_or(HeicMuxError::SizeOverflow("ISO BMFF box"))?;
    let size_u32 = u32::try_from(total_len).map_err(|_| HeicMuxError::SizeOverflow("box size"))?;
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&size_u32.to_be_bytes());
    out.extend_from_slice(&fourcc);
    out.extend_from_slice(payload);
    Ok(out)
}

fn wrap_full_box(
    fourcc: [u8; 4],
    version: u8,
    flags: u32,
    payload: &[u8],
) -> Result<Vec<u8>, HeicMuxError> {
    let total_len = payload
        .len()
        .checked_add(12)
        .ok_or(HeicMuxError::SizeOverflow("ISO BMFF FullBox"))?;
    let size_u32 =
        u32::try_from(total_len).map_err(|_| HeicMuxError::SizeOverflow("FullBox size"))?;
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&size_u32.to_be_bytes());
    out.extend_from_slice(&fourcc);
    let version_flags = (u32::from(version) << 24) | (flags & 0x00FF_FFFF);
    out.extend_from_slice(&version_flags.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{HeicColorInfo, HeicMuxError, mux_annex_b_hevc_to_heic, split_annex_b_nals};

    // Minimal valid HEVC Main Profile 8-bit 4:2:0 Annex-B stream (VPS + SPS + PPS + IDR stub).
    fn sample_hevc_annex_b() -> Vec<u8> {
        let mut bs = Vec::new();
        // VPS (type 32 -> 0x40 0x01)
        bs.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00,
            0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x97, 0x02, 0x40,
        ]);
        // SPS (type 33 -> 0x42 0x01): Main profile (1), level 60, chroma=1 (4:2:0), 8-bit, 256x256
        // RBSP bits after profile_tier_level (12 bytes):
        // sps_id=0 (1), chroma_format_idc=1 (010), pic_w=256 (00000000100000001),
        // pic_h=256 (00000000100000001), conf_win=0 (0), bit_depth_luma=0 (1), bit_depth_chroma=0 (1)
        bs.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90,
            0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x3c, 0xa0, 0x08, 0x08, 0x04, 0x05, 0x80,
        ]);
        // PPS (type 34 -> 0x44 0x01)
        bs.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0xf7, 0xc0, 0xcc, 0x90,
        ]);
        // IDR_W_RADL (type 19 -> 0x26 0x01)
        bs.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x04, 0xf2, 0x0d]);
        bs
    }

    #[test]
    fn muxes_valid_hevc_annex_b_into_well_formed_heic_boxes() -> Result<(), HeicMuxError> {
        let annex_b = sample_hevc_annex_b();
        assert_eq!(split_annex_b_nals(&annex_b).len(), 4);

        let heic = mux_annex_b_hevc_to_heic(&annex_b, 256, 256, HeicColorInfo::BT601_LIMITED)?;
        assert_eq!(heic.get(4..8), Some(b"ftyp".as_slice()));
        assert_eq!(heic.get(8..12), Some(b"heic".as_slice()));

        // Walk top-level ISO BMFF boxes: ftyp, meta, mdat.
        let mut pos = 0_usize;
        let mut top_boxes = Vec::new();
        while pos + 8 <= heic.len() {
            let sz = u32::from_be_bytes([heic[pos], heic[pos + 1], heic[pos + 2], heic[pos + 3]])
                as usize;
            let fourcc = &heic[pos + 4..pos + 8];
            top_boxes.push((fourcc, pos, sz));
            pos += sz;
        }
        assert_eq!(pos, heic.len());
        assert_eq!(top_boxes.len(), 3);
        assert_eq!(top_boxes[0].0, b"ftyp");
        assert_eq!(top_boxes[1].0, b"meta");
        assert_eq!(top_boxes[2].0, b"mdat");

        // Verify mdat contains the 4-byte length-prefixed IDR NAL (6 bytes + 4-byte prefix = 10 bytes).
        let mdat_payload = &heic[top_boxes[2].1 + 8..];
        assert_eq!(mdat_payload.len(), 10);
        assert_eq!(&mdat_payload[..4], &6_u32.to_be_bytes());
        assert_eq!(&mdat_payload[4..6], &[0x26, 0x01]);
        Ok(())
    }

    #[test]
    fn rejects_incomplete_annex_b_bitstreams() {
        let empty = mux_annex_b_hevc_to_heic(&[], 64, 64, HeicColorInfo::BT601_LIMITED);
        assert_eq!(empty, Err(HeicMuxError::MissingNal("VPS")));

        let only_vps = [0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c];
        assert_eq!(
            mux_annex_b_hevc_to_heic(&only_vps, 64, 64, HeicColorInfo::BT601_LIMITED),
            Err(HeicMuxError::MissingNal("SPS"))
        );

        let truncated_nal = [0x00, 0x00, 0x00, 0x01, 0x40];
        assert!(matches!(
            mux_annex_b_hevc_to_heic(&truncated_nal, 64, 64, HeicColorInfo::BT601_LIMITED),
            Err(HeicMuxError::InvalidNal(_))
        ));
    }
}
