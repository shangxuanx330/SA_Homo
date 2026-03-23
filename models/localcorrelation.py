import torch.nn as nn
import torch.nn.functional as F


class LocalCorrelation(nn.Module):
    def __init__(self, radius=4):
        super(LocalCorrelation, self).__init__()
        self.radius = radius

    def forward(self, feat_s, feat_t):
        """
        输入为【bs,(2r+1)*(2r+1), feat_s.h, feats.w】
        """
        B, C, H_s, W_s = feat_s.size()
        B, C, H_t, W_t = feat_t.size()
        r = self.radius
        D = 2 * r + 1

        # Step 1: Add padding
        feat_t_padded = F.pad(feat_t, (r, r, r, r))

        # Step 2: Unfold to extract all patches around each pixel
        # This will give us the patches around each pixel in feat_t
        feat_t_unfold = F.unfold(feat_t_padded, kernel_size=(D, D))

        # Step 3: Rearrange feat_t_unfold to match the size of feat_s
        
        feat_t_unfold = feat_t_unfold.view(B, C, D, D, H_t, W_t)

        # Step 4: Multiply and sum across the channel dimension for correlation
        # Broadcast feat_s to match the size of feat_t_unfold for broadcasting
        feat_s_expanded = feat_s.unsqueeze(2).unsqueeze(2)
        
        # Compute correlation (sum along channel dimension)
        correlation = (feat_s_expanded * feat_t_unfold).sum(dim=1)

        # Step 5: Rearrange the output into the correct shape
        output = correlation.view(B, D * D, H_s, W_s)

        # Normalize the result as in CUDA implementation
        output /= C

        return output