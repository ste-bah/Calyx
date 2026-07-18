use super::*;

#[test]
fn capacity_text_probe_reaches_long_sequence_input() {
    let bytes = capacity_probe_bytes(Modality::Text).unwrap();

    assert!(bytes.len() >= 16 * 1024, "capacity bytes={}", bytes.len());
    assert!(bytes.starts_with(b"Calyx Blackwell warm-load probe"));
}

#[test]
fn capacity_media_probe_remains_a_valid_real_file() {
    let image = capacity_probe_bytes(Modality::Image).unwrap();
    let audio = capacity_probe_bytes(Modality::Audio).unwrap();

    assert!(image.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert!(audio.starts_with(b"RIFF"));
    assert_eq!(&audio[8..12], b"WAVE");
}
