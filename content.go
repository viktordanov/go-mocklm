package main

import (
	"math/rand"
	"strings"
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
