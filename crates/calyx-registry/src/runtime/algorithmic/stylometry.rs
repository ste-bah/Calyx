//! Stylometric pacing features for prose text.
//!
//! Measures HOW prose moves — sentence rhythm, dialogue share, figurative and
//! abstract density, function-word profile — independent of what it says.
//! Semantic lenses cannot distinguish "peak register in every sentence" from
//! properly paced writing in the same voice; this lens exists to make that
//! difference measurable so Ward can gate on it.
//!
//! Encoding: each raw feature is z-scored against frozen reference statistics
//! (English narrative prose; see STYLO_MU/STYLO_SD) and squashed with
//! tanh(z/2). The output vector is [1.0, t_0..t_31]: the constant bias
//! component anchors in-distribution text near the bias axis, so cosine
//! similarity between two in-distribution chunks is ~1 while deviant text
//! rotates away in proportion to how far its statistics sit from the
//! reference. Changing any constant here changes the lens contract; never
//! edit in place — add a new versioned kind instead.

pub(super) const STYLO_FEATURES: usize = 32;
pub(super) const STYLO_DIM: u32 = (STYLO_FEATURES + 1) as u32;

// reference: 2999 narrative chunks (~1000 chars) from the Tolkien corpus,
// computed 2026-07-08 by the mirrored extractor; frozen as
// english-narrative-reference-v1.
const STYLO_MU: [f32; STYLO_FEATURES] = [
    19.681209, 13.679035, 0.300302, 0.211874, 1.626273, 0.335249, 0.029223,
    1.008670, 0.575147, 0.171034, 0.543457, 4.194480, 7.326995, 4.162707,
    3.924473, 1.576381, 2.084592, 2.114717, 1.338819, 1.224729, 1.108903,
    1.216624, 0.763725, 0.584512, 0.716305, 0.752844, 0.568734, 0.824004,
    0.722786, 0.863801, 0.568569, 0.783046,
];
const STYLO_SD: [f32; STYLO_FEATURES] = [
    13.655460, 9.841433, 0.190494, 0.203551, 6.457141, 0.599791, 0.122410,
    0.913297, 0.628115, 0.276006, 0.058918, 0.231822, 2.747561, 2.088762,
    1.709820, 0.975984, 0.964311, 1.173061, 0.797997, 1.199304, 0.842432,
    0.835870, 0.796890, 0.462863, 0.580827, 0.560681, 0.563640, 0.809125,
    0.588741, 0.513144, 0.467224, 0.881808,
];

const FUNCTION_WORDS: [&str; 20] = [
    "the", "of", "and", "a", "to", "in", "that", "he", "it", "was", "his",
    "with", "as", "for", "had", "is", "not", "but", "at", "they",
];
const ABSTRACT_WORDS: [&str; 22] = [
    "hope", "despair", "dominion", "power", "glory", "doom", "fate", "sorrow",
    "splendour", "splendor", "darkness", "light", "memory", "wisdom",
    "justice", "will", "sacrifice", "destiny", "terror", "beauty", "majesty",
    "ruin",
];
const ABSTRACT_SUFFIXES: [&str; 9] = [
    "tion", "sion", "ness", "ment", "ity", "dom", "hood", "ance", "ence",
];

/// Minimum words below which the chunk is too short for stable statistics:
/// it measures as neutral (the bias axis) rather than as noise. A single
/// sentence is NOT neutralised — a 100-word unbroken sentence is precisely
/// the pacing pathology this lens exists to measure.
const MIN_WORDS: usize = 30;
const MIN_SENTENCES: usize = 1;

pub(super) fn stylometry_features(bytes: &[u8]) -> Vec<f32> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = Vec::with_capacity(STYLO_DIM as usize);
    out.push(1.0);
    match raw_features(&text) {
        Some(raw) => {
            for i in 0..STYLO_FEATURES {
                let z = (raw[i] - STYLO_MU[i]) / STYLO_SD[i];
                out.push((z * 0.5).tanh());
            }
        }
        None => out.resize(STYLO_DIM as usize, 0.0),
    }
    out
}

fn is_apostrophe(ch: char) -> bool {
    ch == '\'' || ch == '\u{2019}'
}

fn words_of(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphabetic() {
            cur.extend(ch.to_lowercase());
        } else if is_apostrophe(ch) && !cur.is_empty() {
            cur.push('\'');
        } else if !cur.is_empty() {
            push_word(&mut words, &cur);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        push_word(&mut words, &cur);
    }
    words
}

fn push_word(words: &mut Vec<String>, raw: &str) {
    let trimmed = raw.trim_matches('\'');
    if !trimmed.is_empty() {
        words.push(trimmed.to_string());
    }
}

/// Word counts per sentence; sentences end at runs of . ! ?
fn sentence_lengths(text: &str) -> Vec<usize> {
    let mut sents = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if matches!(ch, '.' | '!' | '?') {
            let n = cur.split_whitespace().count();
            if n > 0 {
                sents.push(n);
            }
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    let n = cur.split_whitespace().count();
    if n > 0 {
        sents.push(n);
    }
    sents
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let (mut sum, mut count) = (0.0_f32, 0_u32);
    for v in values {
        sum += v;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn raw_features(text: &str) -> Option<[f32; STYLO_FEATURES]> {
    let words = words_of(text);
    let sents = sentence_lengths(text);
    if words.len() < MIN_WORDS || sents.len() < MIN_SENTENCES {
        return None;
    }
    let n = words.len() as f32;
    let ns = sents.len() as f32;
    let chars = text.chars().count() as f32;

    let sent_mean = mean(sents.iter().map(|s| *s as f32));
    let sent_var = mean(sents.iter().map(|s| {
        let d = *s as f32 - sent_mean;
        d * d
    }));
    let commas = text.chars().filter(|c| *c == ',').count() as f32;
    let semis = text.chars().filter(|c| matches!(c, ';' | ':')).count() as f32;

    let mut in_quote = false;
    let mut quoted_chars = 0_u32;
    for ch in text.chars() {
        if matches!(ch, '"' | '\u{201C}' | '\u{201D}') {
            in_quote = !in_quote;
        } else if in_quote {
            quoted_chars += 1;
        }
    }

    let mut simile = 0_u32;
    for (i, w) in words.iter().enumerate() {
        if w == "like" {
            simile += 1;
        } else if w == "as"
            && words
                .get(i + 1)
                .is_some_and(|next| next == "if" || next == "though")
        {
            simile += 1;
        }
    }
    let suffixed = words
        .iter()
        .filter(|w| w.len() >= 6 && ABSTRACT_SUFFIXES.iter().any(|s| w.ends_with(s)))
        .count() as f32;
    let abstract_hits = words
        .iter()
        .filter(|w| ABSTRACT_WORDS.contains(&w.as_str()))
        .count() as f32;
    let unique = words
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f32;

    let mut out = [0.0_f32; STYLO_FEATURES];
    out[0] = sent_mean;
    out[1] = sent_var.sqrt();
    out[2] = sents.iter().filter(|s| **s < 8).count() as f32 / ns;
    out[3] = sents.iter().filter(|s| **s > 30).count() as f32 / ns;
    out[4] = commas / ns;
    out[5] = semis / ns;
    out[6] = quoted_chars as f32 / chars.max(1.0);
    out[7] = suffixed / n * 100.0;
    out[8] = abstract_hits / n * 100.0;
    out[9] = simile as f32 / n * 100.0;
    out[10] = unique / n;
    out[11] = mean(words.iter().map(|w| w.chars().count() as f32));
    for (i, fw) in FUNCTION_WORDS.iter().enumerate() {
        out[12 + i] = words.iter().filter(|w| w == fw).count() as f32 / n * 100.0;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLKIENESQUE: &str = "The road went down into the valley, and the \
        hobbits followed it. \"We shall not reach the ford before dark,\" said \
        Strider. \"It is many miles yet.\" Frodo said nothing. His shoulder \
        ached, and the cold of the wound was spreading. They walked on in \
        silence for a while. Sam looked back once. Far behind them, on the \
        crest of the hill, something moved against the sky.";

    const DENSE_PASTICHE: &str = "In the deep of a winter's evening, when the \
        wind from the desolate heights moaned low among the eaves like the \
        lamentation of forgotten kings, the shadow of despair lay upon the \
        hearts of men as a shroud upon the faces of the dead. The splendour of \
        that ancient dominion, which had waxed beneath the majesty of elder \
        days and waned into the sorrow of ruin, pressed down upon the memory \
        of the living as if the very darkness were a burden of judgement, a \
        contrivance of doom and destiny. It choked the hope that yet lingered, \
        veiling the glory that had faded, drowning the wisdom that endured \
        like embers beneath the ash of desolation and the terror of the night, \
        as though the sorrow of the world were a vessel of the majesty of \
        despair, and the ruin of hope a monument to the dominion of darkness.";

    #[test]
    fn output_shape_and_bias() {
        let v = stylometry_features(TOLKIENESQUE.as_bytes());
        assert_eq!(v.len(), STYLO_DIM as usize);
        assert_eq!(v[0], 1.0);
        assert!(v.iter().all(|x| (-1.0..=1.0).contains(x)));
    }

    #[test]
    fn short_input_is_neutral() {
        let v = stylometry_features(b"Too short.");
        assert_eq!(v[0], 1.0);
        assert!(v[1..].iter().all(|x| *x == 0.0));
    }

    #[test]
    fn deterministic() {
        assert_eq!(
            stylometry_features(TOLKIENESQUE.as_bytes()),
            stylometry_features(TOLKIENESQUE.as_bytes())
        );
    }

    #[test]
    fn paced_prose_measures_closer_to_reference_than_dense_pastiche() {
        let cos_to_bias = |v: &[f32]| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v[0] / norm.max(1e-9)
        };
        let paced = cos_to_bias(&stylometry_features(TOLKIENESQUE.as_bytes()));
        let dense = cos_to_bias(&stylometry_features(DENSE_PASTICHE.as_bytes()));
        assert!(
            paced > dense,
            "paced prose should sit nearer the reference axis: paced={paced} dense={dense}"
        );
    }
}
