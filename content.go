package main

import (
	"math/rand"
	"strings"
	"unicode"
)

var wordList = []string{
	"quantum", "neural", "reasoning", "context", "inference",
	"transformer", "attention", "gradient", "embedding", "latent",
	"architecture", "pipeline", "optimization", "convergence", "benchmark",
	"tokenizer", "decoder", "encoder", "layer", "activation",
	"parameter", "training", "validation", "prediction", "classification",
	"generation", "sampling", "probability", "distribution", "entropy",
	"semantic", "syntactic", "morphological", "pragmatic", "discourse",
	"alignment", "calibration", "distillation", "pruning", "quantization",
	"retrieval", "augmentation", "grounding", "hallucination", "factuality",
	"coherence", "fluency", "relevance", "diversity", "consistency",
	"scalability", "efficiency", "throughput", "bandwidth", "capacity",
	"robustness", "generalization", "specialization", "adaptation", "transfer",
	"supervision", "reinforcement", "exploration", "exploitation", "reward",
	"objective", "constraint", "hypothesis", "evaluation", "iteration",
	"computation", "abstraction", "representation", "transformation", "integration",
	"modular", "compositional", "hierarchical", "recursive", "sequential",
	"parallel", "distributed", "centralized", "federated", "collaborative",
	"autonomous", "interactive", "adaptive", "dynamic", "emergent",
	"synthetic", "natural", "hybrid", "multimodal", "cross-domain",
	"innovative", "experimental", "theoretical", "practical", "systematic",
}

func generateWords(n int) []string {
	offset := rand.Intn(len(wordList))
	words := make([]string, n)
	for i := range n {
		words[i] = wordList[(offset+i)%len(wordList)]
	}
	return words
}

// generateDeterministicWords returns a fixed sequence of words starting from index 0.
// Used by deterministic presets so response content is predictable across runs.
func generateDeterministicWords(n int) []string {
	words := make([]string, n)
	for i := range n {
		words[i] = wordList[i%len(wordList)]
	}
	return words
}

// chunkExact slices s losslessly into stream deltas per the Chunking spec.
// The load-bearing property, for every mode and every input (including empty
// and whitespace-only): strings.Join(chunkExact(s, c), "") == s. Chunks are
// UTF-8-clean (never split a rune), so they round-trip through the JSON
// delta marshal — deliberately NO byte-split mode (invalid-UTF-8 testing is
// a transport concern: fragment_split "rune" in sse.go).
//
// Modes: "whole" (one chunk), "runes" (Size runes per chunk, the default),
// "words" (Size whitespace-delimited words per chunk via a prefix-inclusive
// scanner: leading whitespace rides the first chunk, each word carries its
// trailing whitespace, whitespace-only input is a single chunk equal to
// itself). Size <= 0 means whole in every mode.
func chunkExact(s string, c Chunking) []string {
	if s == "" {
		return nil
	}
	if c.Size <= 0 || c.Mode == "whole" {
		return []string{s}
	}
	switch c.Mode {
	case "words":
		return chunkExactWords(s, c.Size)
	default: // "" or "runes"
		return chunkExactRunes(s, c.Size)
	}
}

// chunkExactRunes cuts s into slices of size runes each, never splitting a
// rune.
func chunkExactRunes(s string, size int) []string {
	var chunks []string
	start, n := 0, 0
	for i := range s { // i iterates over rune start offsets
		if n > 0 && n%size == 0 {
			chunks = append(chunks, s[start:i])
			start = i
		}
		n++
	}
	return append(chunks, s[start:])
}

// chunkExactWords cuts s before every size-th word start (a non-whitespace
// rune preceded by whitespace). No byte is ever discarded: leading
// whitespace rides the first chunk, trailing whitespace the last, and a
// whitespace-only string is one chunk equal to itself.
func chunkExactWords(s string, size int) []string {
	var chunks []string
	start, word := 0, 0
	prevSpace := true
	for i, r := range s {
		isSpace := unicode.IsSpace(r)
		if !isSpace && prevSpace {
			// A word starts at i. Cut here only between words — never
			// before the first, so any leading whitespace is kept.
			if word > 0 && word%size == 0 {
				chunks = append(chunks, s[start:i])
				start = i
			}
			word++
		}
		prevSpace = isSpace
	}
	return append(chunks, s[start:])
}

// streamDeltas resolves the per-delta strings for a text stream. The
// exact-output path slices Output.Text losslessly via chunkExact (verbatim
// substrings, no decoration); the generated path keeps the historical word
// decoration (capitalize + trailing period unless content_text pinned the
// words, single-space joins).
func streamDeltas(cfg *ProviderConfig, words []string, exact *ExactOutput) []string {
	if exact != nil {
		return chunkExact(exact.Text, exact.Chunking)
	}
	deltas := make([]string, len(words))
	for i, word := range words {
		token := word
		if cfg.ContentText == "" {
			if i == 0 {
				token = capitalize(token)
			}
			if i == len(words)-1 {
				token += "."
			}
		}
		if i != len(words)-1 {
			token += " "
		}
		deltas[i] = token
	}
	return deltas
}

func joinContent(words []string) string {
	if len(words) == 0 {
		return ""
	}
	s := strings.Join(words, " ")
	// Capitalize first letter
	s = strings.ToUpper(s[:1]) + s[1:]
	s += "."

	return s
}
