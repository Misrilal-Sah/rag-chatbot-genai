"""
Generator Module
Uses HuggingFace Transformers for LLM-based response generation.
Uses FLAN-T5 model (free, no API key required).
"""

from typing import Optional
import torch


class Generator:
    """
    Generates responses using HuggingFace Transformers.
    Default model: google/flan-t5-large (good quality, runs locally)
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: Optional[str] = None
    ):
        """
        Initialize the generator with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading LLM: {model_name} on {device}...")
        print("(This may take a few minutes on first run to download the model)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.model_name = model_name
        
        print(f"✓ Loaded LLM: {model_name}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def generate_rag_response(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate a RAG response given a question and retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context from the knowledge base
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        prompt = self._create_rag_prompt(question, context)
        return self.generate(prompt, max_new_tokens=max_new_tokens)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a prompt for RAG-style question answering."""
        return f"""Answer the following question based on the provided context. 
If the context doesn't contain enough information to answer the question fully, 
say so and provide what information you can based on the context.

Context:
{context}

Question: {question}

Answer:"""


# Alternative generator using smaller model for faster inference
class LightGenerator:
    """
    Lighter generator using FLAN-T5-base for faster inference.
    Trade-off: slightly lower quality responses.
    """
    
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Loading lightweight LLM (flan-t5-base)...")
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✓ Loaded lightweight LLM (device: {self.device})")
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response."""
        # Tokenize with truncation to fit model limits
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=450  # Leave room for generation
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_rag_response(self, question: str, context: str) -> str:
        """Generate a RAG response with context truncation."""
        # Truncate context to prevent token overflow
        # Keep most relevant part (beginning) of each chunk
        max_context_chars = 1200  # Roughly 300 tokens
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        
        # Use a simple, direct prompt format that works well with FLAN-T5
        prompt = f"""Answer the question using the context below.

Context: {context}

Question: {question}

Answer:"""
        
        answer = self.generate(prompt, max_new_tokens=200)
        
        # If answer is too short or seems wrong, try a simpler prompt
        if len(answer) < 10 or answer in ["(iii).", "(i).", "(ii)."]:
            simple_prompt = f"Based on this text: {context[:800]} Answer: {question}"
            answer = self.generate(simple_prompt, max_new_tokens=200)
        
        return answer


def get_generator(use_light: bool = False):
    """
    Factory function to get a generator.
    
    Args:
        use_light: If True, use the lighter/faster model
        
    Returns:
        Generator or LightGenerator instance
    """
    if use_light:
        return LightGenerator()
    return Generator()


if __name__ == "__main__":
    # Test the generator
    print("Testing generator...")
    
    gen = LightGenerator()  # Use lighter model for testing
    
    context = """
    RAG (Retrieval-Augmented Generation) is a technique that combines 
    retrieval-based methods with generative models. It first retrieves 
    relevant documents from a knowledge base, then uses an LLM to generate 
    responses based on both the query and retrieved context.
    """
    
    question = "What is RAG?"
    
    answer = gen.generate_rag_response(question, context)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
