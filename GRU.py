import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(GRU,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #Update Gate
        self.xu=nn.Linear(input_dim,hidden_dim)
        self.hu=nn.Linear(hidden_dim,hidden_dim,bias=False)

        #Reset Gate
        self.xr=nn.Linear(input_dim,hidden_dim)
        self.hr=nn.Linear(hidden_dim,hidden_dim,bias=False)

        #Candidate Hidden Memory
        self.xn=nn.Linear(input_dim,hidden_dim)
        self.hn=nn.Linear(hidden_dim,hidden_dim,bias=False)


    def forward(self,x,h_prev):
        z_t=torch.sigmoid(self.xu(x)+self.hu(h_prev))
        r_t=torch.sigmoid(self.xr(x)+self.hr(h_prev))
        m_f=r_t*h_prev
        c_h=torch.tanh(self.xn(x)+self.hn(m_f))
        h_t=(1-z_t)*h_prev+z_t*c_h

        return h_t


if __name__ == "__main__":
    print("=" * 60)
    print("GRU TEST")
    print("=" * 60)

    # RSSM uses GRU(input_dim = s_dim + a_dim = 1024 + 17 = 1041, hidden = 600)
    input_dim = 1041
    hidden_dim = 600
    gru = GRU(input_dim, hidden_dim)

    B = 8
    x = torch.randn(B, input_dim)
    h_prev = torch.zeros(B, hidden_dim)

    # Single step
    h = gru(x, h_prev)
    print(f"\nSingle step:")
    print(f"  x:       {x.shape}      (expect (8, 1041))")
    print(f"  h_prev:  {h_prev.shape}  (expect (8, 600))")
    print(f"  h_out:   {h.shape}       (expect (8, 600))")
    print(f"  finite:  {torch.isfinite(h).all().item()}")

    # 50-step rollout: feed h back in, confirm it stays finite and bounded.
    h = torch.zeros(B, hidden_dim)
    for t in range(50):
        x_t = torch.randn(B, input_dim)
        h = gru(x_t, h)
    print(f"\n50-step rollout:")
    print(f"  h_out:   {h.shape}")
    print(f"  finite:  {torch.isfinite(h).all().item()}  (must be True -- no NaN/inf)")
    print(f"  range:   [{h.min():.3f}, {h.max():.3f}]  "
          f"(tanh-bounded candidate keeps this sane)")

    # Gradient check: loss on h should backprop to all gate weights.
    h = torch.zeros(B, hidden_dim)
    x_t = torch.randn(B, input_dim)
    out = gru(x_t, h)
    out.sum().backward()
    grads_exist = all(p.grad is not None for p in gru.parameters())
    print(f"\nGradient check:")
    print(f"  all params have grad: {grads_exist}  (expect True)")

    n_params = sum(p.numel() for p in gru.parameters())
    print(f"\nTotal params: {n_params:,}")

    print("\n" + "=" * 60)
    print("GRU WORKS")
    print("=" * 60)