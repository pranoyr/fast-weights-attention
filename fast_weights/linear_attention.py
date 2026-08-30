import torch
import torch.nn as nn
from einops import rearrange, einsum

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.norm = nn.LayerNorm(dim)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def phi(self, x):
        """
        Feature map function to ensure positive keys and queries.
        Uses ELU + 1 which is a standard choice for fast-weight / linear attention.
        """
        return torch.nn.functional.elu(x) + 1

    def forward(self, x):
        """Standard bidirectional forward pass."""
     
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Apply phi feature map to Q and K
        q = self.phi(q)
        k = self.phi(k)

        # FAST WEIGHT MEMORY (Update Rule)
        memory = einsum(k, v, 'b h n d_k, b h n d_v -> b h d_k d_v')

        # RETRIEVAL RULE
        out = einsum(q, memory, 'b h n d_k, b h d_k d_v -> b h n d_v')

        # Rearrange heads back and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward_step(self, x_chunk, memory_state=None):
        """
        RNN-like step function. Processes a chunk or token, updates the memory,
        and returns the output and the new memory state.
        """
        x_chunk = self.norm(x_chunk)

        qkv = self.to_qkv(x_chunk).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = self.phi(q)
        k = self.phi(k)

        # Update Memory State with current chunk
        chunk_memory = einsum(k, v, 'b h n d_k, b h n d_v -> b h d_k d_v')
        if memory_state is not None:
            memory_state = memory_state + chunk_memory
        else:
            memory_state = chunk_memory

        # 2. Retrieve output from Memory
        out = einsum(q, memory_state, 'b h n d_k, b h d_k d_v -> b h n d_v')
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), memory_state


if __name__ == "__main__":
    # Sample usage demonstrating the LinearAttention module
    batch_size = 2
    seq_len = 1000  # A very long sequence
    dim = 256
    heads = 8
    
    attn = LinearAttention(dim=dim, heads=heads)
    
    # Simulate a streaming or extremely long input
    sample_input = torch.randn(batch_size, seq_len, dim)
    
 
    chunk_size = 100
    memory_state = None
    output_chunks = []
    
    print("Processing long sequence chunk by chunk...")
    for i in range(0, seq_len, chunk_size):
        # Grab a chunk of the sequence
        chunk = sample_input[:, i:i+chunk_size, :]
        
        # Pass the chunk and the previous memory state
        out_chunk, memory_state = attn.forward_step(chunk, memory_state)
        output_chunks.append(out_chunk)
        
        print(f"Processed chunk {i//chunk_size + 1}, Memory shape: {memory_state.shape}")
        
    final_output = torch.cat(output_chunks, dim=1)
    
    print(f"\nFinal output shape: {final_output.shape}")
    print(f"Does the final output shape match the input shape? {sample_input.shape == final_output.shape}")
