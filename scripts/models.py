import torch 
import torch.nn as nn
import torch.nn.functional as F

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super(SiameseNetwork, self).__init__()

        self.fc = nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2))

        self.fc5 = nn.Linear(128, output_dim)

        #self.fc5 = nn.Linear(16 * 2, output_dim)  # Output layer for similarity judgements

    def forward_one(self, x):
        return self.fc(x)

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        # pdb.set_trace()
        # Combine both outputs by concatenation
        # combined = torch.concat((out1, out2), dim=1) # concatenate embeddings
        combined = torch.sub(out1, out2)  # maybe torch.abs(out1 - out2)
        output = self.fc5(combined)                  # Outputs raw logits
        return output

class NystromAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, k=64):
        super(NystromAttention, self).__init__()
        self.num_heads = num_heads
        self.k = k  # Number of landmark points
        self.scale = (input_dim // num_heads) ** -0.5
        
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)
        self.linear_out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)  # Get batch size

        # Linear transformations
        q = self.linear_q(x).view(batch_size, self.num_heads, -1)
        k = self.linear_k(x).view(batch_size, self.num_heads, -1)
        v = self.linear_v(x).view(batch_size, self.num_heads, -1)

        # Sample k landmark points from k
        indices = torch.randint(0, batch_size, (self.k,))
        k_landmarks = k[indices]  # Shape: (k, num_heads, head_dim)
        v_landmarks = v[indices]  # Shape: (k, num_heads, head_dim)

        # Compute attention scores
        attn_scores = torch.einsum('bhd,khd->bhk', q, k_landmarks) * self.scale  # Change here
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.einsum('bhk,khd->bhd', attn_weights, v_landmarks)

        # Reshape or mean pool to get to (batch_size, input_dim)
        attn_output = attn_output.reshape(batch_size, -1)  # Flatten if necessary or use mean pooling
        return self.linear_out(attn_output)  # Pass through final linear layer
    

class SiameseNetworkWithNystrom(nn.Module):
    def __init__(self, input_dim=768, output_dim=5):
        super(SiameseNetworkWithNystrom, self).__init__()

        # Use Nyström Attention layer instead of feedforward layers
        self.attention_layer = NystromAttention(input_dim=input_dim)

        # Final output layer for classification or regression tasks
        self.fc5 = nn.Linear(input_dim, output_dim)

    def forward_one(self, x):
        return self.attention_layer(x)

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)  # Process first embedding
        out2 = self.forward_one(input2)  # Process second embedding

        combined = torch.sub(out1, out2)  # Subtract outputs for similarity measure
        output = self.fc5(combined)  # Outputs raw logits
        return output