//! Bounded, append-only text chunks for the exact admitted tokenizer profiles.
//!
//! This text-only API grants no numerical context or app output capability.
//! ByteLevel retains at most three incomplete UTF-8 bytes. Mistral must retain
//! a whole consecutive fallback-byte run: one later invalid byte can rewrite it
//! entirely. That run is bounded by the admitted response-token count and is
//! decoded once at a normal-token boundary or finish, never per-byte prefix.
use super::{Codec, Profile, invalid};
use std::io;
use tokenizers::Tokenizer;

pub const MAX_STREAM_TOKENS: usize = 131072;
pub const MAX_STREAM_TEXT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_STREAM_BATCH: usize = 128;
const MAX_TOKEN_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub struct ResponseLimits {
    tokens: usize,
    text_bytes: usize,
}
impl ResponseLimits {
    pub fn new(tokens: usize, text_bytes: usize) -> io::Result<Self> {
        if !(1..=MAX_STREAM_TOKENS).contains(&tokens)
            || !(1..=MAX_STREAM_TEXT_BYTES).contains(&text_bytes)
        {
            return Err(invalid("Response text limits exceed the streaming bounds"));
        }
        Ok(Self { tokens, text_bytes })
    }
}

#[derive(Clone, Copy)]
enum Decoder {
    ByteLevel,
    ByteFallback,
}

/// Chunks borrow this stream until its next call. Copy/consume only the new
/// chunk; the stream does not retain or clone the complete emitted response.
/// Any failed mutation poisons the stream; create a fresh one and replay IDs.
/// Finish is required, including cancellation/early stop, to flush pending text.
/// Special IDs count toward the token limit but emit no text. Unknown IDs below
/// the profile vocabulary bound are skipped, matching the original tokenizer.
pub struct ResponseStream<'codec> {
    tokenizer: &'codec Tokenizer,
    state: ResponseState,
}

// The codec mediates all access to its tokenizer. This owned state has no
// self-reference and cannot be paired with arbitrary tokenizers by callers.
struct ResponseState {
    vocab: u32,
    limits: ResponseLimits,
    decoder: Decoder,
    special_ids: Vec<u32>,
    tokens: usize,
    text_bytes: usize,
    delta: String,
    utf8: [u8; 4],
    utf8_len: usize,
    fallback: Vec<u8>,
    at_start: bool,
    finished: bool,
    failed: bool,
}
impl Codec {
    pub fn response_stream(&self, limits: ResponseLimits) -> ResponseStream<'_> {
        // Selection and token inventory are cold. Numerical kernels see neither.
        let decoder = match self.profile {
            Profile::Smol | Profile::Qwen | Profile::Qwen15 | Profile::Qwen3 | Profile::Llama3 => {
                Decoder::ByteLevel
            }
            Profile::Mistral => Decoder::ByteFallback,
        };
        ResponseStream::new(&self.tokenizer, self.profile.vocab(), decoder, true, limits)
    }
}
impl<'codec> ResponseStream<'codec> {
    pub(super) fn byte_fallback(
        tokenizer: &'codec Tokenizer,
        vocab: u32,
        strip_leading_space: bool,
        limits: ResponseLimits,
    ) -> Self {
        Self::new(
            tokenizer,
            vocab,
            Decoder::ByteFallback,
            strip_leading_space,
            limits,
        )
    }
    fn new(
        tokenizer: &'codec Tokenizer,
        vocab: u32,
        decoder: Decoder,
        strip_leading_space: bool,
        limits: ResponseLimits,
    ) -> Self {
        let mut special_ids: Vec<_> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect();
        special_ids.sort_unstable();
        // Worst case: replacement UTF-8 per input byte, plus a retained fallback
        // run. Limits bound arithmetic; this reserves delivery buffers once.
        // id_to_token/Replace still allocate temporary text per consumed ID.
        let delta_capacity = limits
            .text_bytes
            .min(3 * (MAX_STREAM_BATCH * MAX_TOKEN_BYTES + limits.tokens) + 3);
        Self {
            tokenizer,
            state: ResponseState {
                vocab,
                limits,
                decoder,
                special_ids,
                tokens: 0,
                text_bytes: 0,
                delta: String::with_capacity(delta_capacity),
                utf8: [0; 4],
                utf8_len: 0,
                fallback: Vec::with_capacity(if matches!(decoder, Decoder::ByteFallback) {
                    limits.tokens
                } else {
                    0
                }),
                at_start: strip_leading_space,
                finished: false,
                failed: false,
            },
        }
    }
}
impl ResponseStream<'_> {
    pub fn consumed_tokens(&self) -> usize {
        self.state.consumed_tokens()
    }
    pub fn emitted_bytes(&self) -> usize {
        self.state.emitted_bytes()
    }
    pub fn is_finished(&self) -> bool {
        self.state.is_finished()
    }
    pub fn is_failed(&self) -> bool {
        self.state.is_failed()
    }
    pub fn push(&mut self, ids: &[u32]) -> io::Result<&str> {
        self.state.push(self.tokenizer, ids)
    }
    pub fn finish(&mut self) -> io::Result<&str> {
        self.state.finish()
    }
    /// Owned capacities plus State/facade metadata, excluding the borrowed codec.
    /// Per-token temporary text and caller-retained chunks need separate allowance.
    pub fn allocated_bytes(&self) -> io::Result<usize> {
        let special = self
            .state
            .special_ids
            .capacity()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| invalid("Response storage accounting overflow"))?;
        [
            self.state.delta.capacity(),
            self.state.fallback.capacity(),
            special,
        ]
        .into_iter()
        .try_fold(std::mem::size_of::<Self>(), |sum, n| {
            sum.checked_add(n)
                .ok_or_else(|| invalid("Response storage accounting overflow"))
        })
    }
}
impl ResponseState {
    pub fn consumed_tokens(&self) -> usize {
        self.tokens
    }
    pub fn emitted_bytes(&self) -> usize {
        self.text_bytes
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    pub fn is_failed(&self) -> bool {
        self.failed
    }
    pub fn push(&mut self, tokenizer: &Tokenizer, ids: &[u32]) -> io::Result<&str> {
        self.delta.clear();
        let previous = (self.tokens, self.text_bytes);
        if let Err(error) = self.push_inner(tokenizer, ids) {
            self.failed = true;
            self.delta.clear();
            // Nothing from this failed call escaped. The decoder state is
            // poisoned, but counters still describe successfully returned text.
            (self.tokens, self.text_bytes) = previous;
            return Err(error);
        }
        Ok(&self.delta)
    }
    fn push_inner(&mut self, tokenizer: &Tokenizer, ids: &[u32]) -> io::Result<()> {
        self.active()?;
        let count = self
            .tokens
            .checked_add(ids.len())
            .ok_or_else(|| invalid("Response token count overflow"))?;
        if ids.len() > MAX_STREAM_BATCH
            || count > self.limits.tokens
            || ids.iter().any(|&id| id >= self.vocab)
        {
            return Err(invalid("Response batch exceeds admitted token bounds"));
        }
        // Check the entire batch before consuming it. Failure still poisons the
        // text stream; it never mutates a numerical session or sampled history.
        self.tokens = count;
        match self.decoder {
            Decoder::ByteLevel => {
                for &id in ids {
                    if let Some(token) = self.token(tokenizer, id)? {
                        // Upstream falls back to the *whole* token's UTF-8 if
                        // any character is outside the ByteLevel alphabet.
                        if token.chars().all(|c| byte_value(c).is_some()) {
                            for c in token.chars() {
                                self.byte(byte_value(c).expect("Checked ByteLevel alphabet"))?;
                            }
                        } else {
                            for byte in token.bytes() {
                                self.byte(byte)?;
                            }
                        }
                    }
                }
            }
            Decoder::ByteFallback => {
                for &id in ids {
                    if let Some(token) = self.token(tokenizer, id)? {
                        let token = token.replace('▁', " ");
                        if let Some(byte) = fallback_byte(&token) {
                            // At most one byte per consumed fallback ID.
                            self.fallback.push(byte);
                        } else {
                            self.flush_fallback()?;
                            self.fallback_text(&token)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn token(&self, tokenizer: &Tokenizer, id: u32) -> io::Result<Option<String>> {
        if self.special_ids.binary_search(&id).is_ok() {
            return Ok(None);
        }
        let token = tokenizer.id_to_token(id);
        if token
            .as_ref()
            .is_some_and(|token| token.len() > MAX_TOKEN_BYTES)
        {
            return Err(invalid("Response token exceeds streaming byte bound"));
        }
        Ok(token)
    }
    pub fn finish(&mut self) -> io::Result<&str> {
        self.delta.clear();
        let result = self.active().and_then(|()| match self.decoder {
            Decoder::ByteLevel => {
                if self.utf8_len != 0 {
                    self.text("�")?;
                    self.utf8_len = 0;
                }
                Ok(())
            }
            Decoder::ByteFallback => self.flush_fallback(),
        });
        if let Err(error) = result {
            self.failed = true;
            self.delta.clear();
            return Err(error);
        }
        self.finished = true;
        Ok(&self.delta)
    }
    fn active(&self) -> io::Result<()> {
        if self.failed || self.finished {
            return Err(invalid(
                "Response stream is terminal; create a fresh stream",
            ));
        }
        Ok(())
    }
    fn text(&mut self, text: &str) -> io::Result<()> {
        let size = self
            .text_bytes
            .checked_add(text.len())
            .ok_or_else(|| invalid("Response byte count overflow"))?;
        if size > self.limits.text_bytes {
            return Err(invalid("Response exceeds admitted text bytes"));
        }
        self.delta.push_str(text);
        self.text_bytes = size;
        Ok(())
    }
    fn byte(&mut self, byte: u8) -> io::Result<()> {
        // An incomplete sequence has <=3 bytes. Appending one is in bounds.
        self.utf8[self.utf8_len] = byte;
        self.utf8_len += 1;
        loop {
            // Copy four bounded bytes so text publication does not alias state.
            let bytes = self.utf8;
            match std::str::from_utf8(&bytes[..self.utf8_len]) {
                Ok(text) => {
                    self.text(text)?;
                    self.utf8_len = 0;
                    return Ok(());
                }
                Err(error) => {
                    // Complete scalars were already emitted, hence no valid
                    // prefix precedes an error in this at-most-four-byte carry.
                    debug_assert_eq!(error.valid_up_to(), 0);
                    let Some(count) = error.error_len() else {
                        return Ok(());
                    };
                    self.text("�")?;
                    self.utf8.copy_within(count..self.utf8_len, 0);
                    self.utf8_len -= count;
                }
            }
        }
    }
    fn fallback_text(&mut self, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let text = if self.at_start {
            self.at_start = false;
            text.strip_prefix(' ').unwrap_or(text)
        } else {
            text
        };
        self.text(text)
    }
    fn flush_fallback(&mut self) -> io::Result<()> {
        if self.fallback.is_empty() {
            return Ok(());
        }
        // Move ownership out briefly to avoid cloning the run for publication.
        // Restore its reserved storage even on a terminal byte-budget failure.
        let mut run = std::mem::take(&mut self.fallback);
        let result = match std::str::from_utf8(&run) {
            Ok(text) => self.fallback_text(text),
            Err(_) => {
                self.at_start = false;
                let bytes = run.len() * "�".len();
                if bytes > self.limits.text_bytes - self.text_bytes {
                    Err(invalid("Response exceeds admitted text bytes"))
                } else {
                    self.delta.extend(std::iter::repeat_n('�', run.len()));
                    self.text_bytes += bytes;
                    Ok(())
                }
            }
        };
        run.clear();
        self.fallback = run;
        result
    }
}
fn fallback_byte(token: &str) -> Option<u8> {
    (token.len() == 6 && token.starts_with("<0x") && token.ends_with('>'))
        .then(|| u8::from_str_radix(&token[3..5], 16).ok())
        .flatten()
}
fn byte_value(c: char) -> Option<u8> {
    let c = u32::from(c);
    if matches!(c, 33..=126 | 161..=172 | 174..=255) {
        return Some(c as u8);
    }
    // Missing direct bytes map in ascending byte order to U+0100 onward.
    match c {
        256..=288 => Some((c - 256) as u8),
        289..=322 => Some((c - 289 + 127) as u8),
        323 => Some(173),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use tokenizers::Tokenizer;

    fn fixture(profile: &str) -> Codec {
        let (profile, raw) = match profile {
            "smol" => (
                Profile::Smol,
                include_bytes!("../../tests/fixtures/chat-stream-v1/smol-tokenizer.json")
                    .as_slice(),
            ),
            "qwen" => (
                Profile::Qwen,
                include_bytes!("../../tests/fixtures/chat-stream-v1/qwen-tokenizer.json")
                    .as_slice(),
            ),
            "mistral" => (
                Profile::Mistral,
                include_bytes!("../../tests/fixtures/chat-stream-v1/mistral-tokenizer.json")
                    .as_slice(),
            ),
            _ => panic!("Unknown fixture"),
        };
        Codec {
            tokenizer: Tokenizer::from_bytes(raw).unwrap(),
            profile,
            special_literals: Vec::new(),
            content_identity: None,
            dense_content_identity: None,
        }
    }
    fn reference() -> Value {
        serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/chat-stream-v1/reference.json"
        ))
        .unwrap()
    }
    fn ids(value: &Value) -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
            .collect()
    }
    #[test]
    fn frozen_complete_prefixes_and_terminal_text_match_for_every_partition() {
        let reference = reference();
        let mut cases = 0;
        for (name, profile) in reference["profiles"].as_object().unwrap() {
            let codec = fixture(name);
            for case in profile["cases"].as_array().unwrap() {
                let input = ids(&case["ids"]);
                let prefixes = case["prefixes"].as_array().unwrap();
                let expected = prefixes.last().unwrap().as_str().unwrap();
                for batch in [1, 2, 7, MAX_STREAM_BATCH] {
                    let mut stream = codec
                        .response_stream(ResponseLimits::new(input.len().max(1), 4096).unwrap());
                    let mut visible = String::new();
                    for chunk in input.chunks(batch) {
                        visible.push_str(stream.push(chunk).unwrap());
                        let prefix = prefixes[stream.consumed_tokens()].as_str().unwrap();
                        assert!(
                            prefix.starts_with(&visible),
                            "{name}/{} complete-prefix divergence",
                            case["name"]
                        );
                        assert!(
                            expected.starts_with(&visible),
                            "Published text was later rewritten"
                        );
                        assert!(
                            stream.state.utf8_len <= 3
                                && stream.state.fallback.len() <= input.len()
                        );
                    }
                    visible.push_str(stream.finish().unwrap());
                    assert_eq!(visible, expected, "{name}/{} batch{batch}", case["name"]);
                    assert_eq!(stream.consumed_tokens(), input.len());
                    assert_eq!(stream.emitted_bytes(), visible.len());
                    assert!(stream.is_finished() && !stream.is_failed());
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 33);
    }
    #[test]
    fn frozen_131072_id_outputs_hash_without_retaining_whole_response() {
        let reference = reference();
        let mut cases = 0;
        for (name, profile) in reference["profiles"].as_object().unwrap() {
            let codec = fixture(name);
            for case in profile["long_cases"].as_array().unwrap() {
                let pattern = ids(&case["repeat_ids"]);
                let mut input = pattern.repeat(case["repeat_count"].as_u64().unwrap() as usize);
                input.extend(ids(&case["tail_ids"]));
                assert_eq!(input.len(), MAX_STREAM_TOKENS);
                for batch in [1, MAX_STREAM_BATCH] {
                    let mut stream = codec.response_stream(
                        ResponseLimits::new(MAX_STREAM_TOKENS, MAX_STREAM_TEXT_BYTES).unwrap(),
                    );
                    let capacities = (
                        stream.state.delta.capacity(),
                        stream.state.fallback.capacity(),
                        stream.state.special_ids.capacity(),
                    );
                    let mut hash = Sha256::new();
                    for chunk in input.chunks(batch) {
                        hash.update(stream.push(chunk).unwrap().as_bytes());
                        assert!(
                            stream.state.utf8_len <= 3
                                && stream.state.fallback.len() <= MAX_STREAM_TOKENS
                        );
                    }
                    hash.update(stream.finish().unwrap().as_bytes());
                    assert_eq!(
                        format!("{:x}", hash.finalize()),
                        case["text_sha256"].as_str().unwrap()
                    );
                    assert_eq!(
                        stream.emitted_bytes(),
                        case["utf8_bytes"].as_u64().unwrap() as usize
                    );
                    assert_eq!(stream.consumed_tokens(), MAX_STREAM_TOKENS);
                    assert_eq!(
                        capacities,
                        (
                            stream.state.delta.capacity(),
                            stream.state.fallback.capacity(),
                            stream.state.special_ids.capacity()
                        )
                    );
                    assert!(stream.finish().is_err() && stream.is_failed());
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 8);
    }
    #[test]
    fn failures_poison_and_require_fresh_stream_without_changing_old_decode() {
        let codec = fixture("qwen");
        assert!(ResponseLimits::new(0, 1).is_err());
        assert!(ResponseLimits::new(1, 0).is_err());
        assert!(ResponseLimits::new(MAX_STREAM_TOKENS + 1, 1).is_err());
        assert!(ResponseLimits::new(1, MAX_STREAM_TEXT_BYTES + 1).is_err());
        for bad in [
            vec![codec.profile.vocab()],
            vec![32; MAX_STREAM_BATCH + 1],
            vec![32; 3],
        ] {
            let mut stream = codec.response_stream(ResponseLimits::new(2, 10).unwrap());
            assert!(stream.push(&bad).is_err() && stream.is_failed());
            assert_eq!(stream.consumed_tokens(), 0);
            assert!(stream.push(&[32]).is_err() && stream.finish().is_err());
        }
        let mut stream = codec.response_stream(ResponseLimits::new(2, 1).unwrap());
        assert!(stream.push(&[32, 32]).is_err() && stream.is_failed());
        assert_eq!((stream.consumed_tokens(), stream.emitted_bytes()), (0, 0));
        let mut stream = codec.response_stream(ResponseLimits::new(2, 1).unwrap());
        assert_eq!(stream.push(&[32]).unwrap(), "A");
        assert!(stream.push(&[32]).is_err() && stream.is_failed());
        assert_eq!(stream.emitted_bytes(), 1);
        assert!(stream.finish().is_err());
        let mut stream = codec.response_stream(ResponseLimits::new(1, 3).unwrap());
        assert_eq!(stream.push(&[151645]).unwrap(), "");
        assert_eq!(stream.consumed_tokens(), 1);
        assert_eq!(stream.finish().unwrap(), "");
        assert!(stream.push(&[]).is_err());
        // Existing foreign-buffer whole-prefix decoding retains its short cap.
        assert_eq!(codec.decode(&[32, 32]).unwrap(), "AA");
        assert!(codec.decode(&vec![32; 257]).is_err());
    }
    #[test]
    fn bounded_utf8_state_matches_lossy_decoder_for_byte_pairs_and_edge_triples() {
        let codec = fixture("qwen");
        fn check(codec: &Codec, bytes: &[u8]) {
            let mut stream = codec.response_stream(ResponseLimits::new(1, 32).unwrap());
            for &byte in bytes {
                stream.state.byte(byte).unwrap();
                assert!(stream.state.utf8_len <= 3);
            }
            let mut text = stream.state.delta.clone();
            text.push_str(stream.finish().unwrap());
            assert_eq!(text, String::from_utf8_lossy(bytes), "{bytes:?}");
        }
        for first in 0..=255 {
            for second in 0..=255 {
                check(&codec, &[first, second]);
            }
        }
        let mut seed = 0x1234_abcd_u32;
        for _ in 0..10000 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            check(&codec, &seed.to_le_bytes());
        }
        for bytes in [
            &[0xe0, 0x80, 0x80][..],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0xf0, 0x90, 0x80, 0x80],
            &[0xef, 0xbf, 0xbd],
        ] {
            check(&codec, bytes);
        }
    }

    #[test]
    #[ignore = "requires the three exact locally acquired tokenizer/config snapshots"]
    fn real_pinned_tokenizers_match_all_vocab_ids_and_frozen_witnesses() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let reference = reference();
        let mut checked = 0;
        let mut maximum = 0;
        for (name, folder) in [
            ("smol", "MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8"),
            ("qwen", "MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c"),
            ("mistral", "MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87"),
        ] {
            let codec = Codec::load(&root.join(folder)).unwrap();
            let vocabulary = codec.tokenizer.get_vocab(true);
            for token in vocabulary.keys() {
                maximum = maximum.max(token.len());
                assert!(token.len() <= MAX_TOKEN_BYTES);
            }
            let mut all: Vec<u32> = vocabulary.into_values().collect();
            all.sort_unstable();
            all.dedup();
            checked += all.len();
            let mut inputs: Vec<Vec<u32>> = all.chunks(65536).map(<[u32]>::to_vec).collect();
            for case in reference["profiles"][name]["cases"].as_array().unwrap() {
                inputs.push(ids(&case["ids"]));
            }
            for case in reference["profiles"][name]["long_cases"]
                .as_array()
                .unwrap()
            {
                let mut input = ids(&case["repeat_ids"])
                    .repeat(case["repeat_count"].as_u64().unwrap() as usize);
                input.extend(ids(&case["tail_ids"]));
                inputs.push(input);
            }
            for input in inputs {
                let expected = codec.tokenizer.decode(&input, true).unwrap();
                let mut stream = codec.response_stream(
                    ResponseLimits::new(input.len().max(1), MAX_STREAM_TEXT_BYTES).unwrap(),
                );
                let mut text = String::new();
                for chunk in input.chunks(MAX_STREAM_BATCH) {
                    text.push_str(stream.push(chunk).unwrap());
                    assert!(expected.starts_with(&text));
                }
                text.push_str(stream.finish().unwrap());
                assert_eq!(text, expected, "Real tokenizer {name}");
            }
        }
        println!(
            "CHAT_STREAM_REAL {}",
            serde_json::json!({"profiles":3,"vocab_ids_checked":checked,"maximum_raw_token_bytes":maximum,"short_cases":33,"long_cases":8,"passed":true})
        );
    }

    #[test]
    fn byte_budget_includes_terminal_replacement_and_retained_mistral_run() {
        for name in ["qwen", "mistral"] {
            let codec = fixture(name);
            let reference = reference();
            let case = reference["profiles"][name]["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == "unfinished_globe")
                .unwrap();
            let mut stream = codec.response_stream(ResponseLimits::new(4, 1).unwrap());
            assert_eq!(stream.push(&ids(&case["ids"])).unwrap(), "");
            assert!(stream.finish().is_err() && stream.is_failed());
            assert_eq!(stream.emitted_bytes(), 0);
        }
    }
    #[test]
    fn absent_bounded_ids_skip_and_whole_token_bytelevel_fallback_is_preserved() {
        let mut codec = fixture("qwen");
        let mut raw: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/chat-stream-v1/qwen-tokenizer.json"
        ))
        .unwrap();
        raw["model"]["vocab"]["Ġ中"] = Value::from(90000);
        codec.tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&raw).unwrap()).unwrap();
        let mut stream = codec.response_stream(ResponseLimits::new(2, 64).unwrap());
        assert_eq!(stream.push(&[151935]).unwrap(), "");
        // One unmapped character makes upstream use the whole token literally,
        // preserving Ġ rather than mapping that first character to a space.
        assert_eq!(stream.push(&[90000]).unwrap(), "Ġ中");
        assert_eq!(stream.finish().unwrap(), "");
        assert_eq!(
            codec.tokenizer.decode(&[151935, 90000], true).unwrap(),
            "Ġ中"
        );
    }
}
