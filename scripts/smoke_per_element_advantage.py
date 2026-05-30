"""Shape/runtime smoke test for the per-element advantage rewrite.

Constructs PolicyNetwork + ValueNetwork on dummy data, exercises both the
legacy scalar path and the new [B, R] tensor path, and confirms .backward()
runs and gradients arrive at the policy parameters.

Run: python scripts/smoke_per_element_advantage.py
"""

import torch
from capn import PolicyNetwork, ValueNetwork


def main():
    torch.manual_seed(0)
    B, R, S = 8, 3, 43  # batch, relations, state_dim — matches Yelp Row 4

    policy = PolicyNetwork(state_dim=S, hidden_dim=64, num_relations=R).train()
    critic = ValueNetwork(state_dim=S, hidden_dim=64).train()

    # Per-relation dummy states + forward pass
    policy.reset_episode()
    states, log_probs_list = [], []
    for r in range(R):
        s = torch.randn(B, S)
        thresh, lp = policy(s, r, deterministic=False)
        policy.store_action(lp, thresh)
        states.append(s)
        log_probs_list.append(lp)
    assert all(lp.shape == (B,) for lp in log_probs_list), "log_probs must be [B] per relation"

    # Per-node, per-relation V(s) -> [B, R]
    values = torch.cat([critic(s) for s in states], dim=1)
    assert values.shape == (B, R), f"expected [B,R], got {values.shape}"

    # New tensor-advantage path
    raw_reward = 0.42
    advantage = raw_reward - values.detach()
    assert advantage.shape == (B, R)
    loss_tensor = policy.get_policy_loss(advantage)
    assert loss_tensor.dim() == 0, "policy loss must be scalar"
    loss_tensor.backward()
    grad_count = sum(1 for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0, "policy got no gradient"

    # Legacy scalar-advantage path (unchanged behaviour)
    policy.zero_grad()
    policy.reset_episode()
    for r in range(R):
        s = torch.randn(B, S)
        thresh, lp = policy(s, r, deterministic=False)
        policy.store_action(lp, thresh)
    loss_scalar = policy.get_policy_loss(0.42)
    assert loss_scalar.dim() == 0
    loss_scalar.backward()

    print(f"OK  tensor-path loss={loss_tensor.item():+.4f}  "
          f"scalar-path loss={loss_scalar.item():+.4f}  "
          f"params with grad={grad_count}")


if __name__ == "__main__":
    main()
