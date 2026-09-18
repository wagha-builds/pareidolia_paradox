import torch
from src.models import AdversarialBackbone, GradientReversal, SmallCNN, predict_proba


def test_gradient_reversal_backward():
    """Verify GRL reverses and scales the gradient during backward pass."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    lam = 0.5
    y = GradientReversal.apply(x, lam)
    loss = (y * torch.tensor([2.0, 4.0, 6.0])).sum()
    loss.backward()

    # dy/dx should be -lam * grad_output = -0.5 * [2, 4, 6] = [-1, -2, -3]
    expected_grad = torch.tensor([-1.0, -2.0, -3.0])
    assert torch.allclose(x.grad, expected_grad)


def test_adversarial_backbone_forward_train_and_eval():
    """Verify AdversarialBackbone returns main logits and az logits during training, and main logits during eval."""
    base = SmallCNN(in_chans=1)
    model = AdversarialBackbone(
        base,
        feature_dim=64,
        num_classes=2,
        num_az_classes=4,
        adv_lambda=0.5,
        film_hidden=32,
    )

    dummy_img = torch.randn(4, 1, 256, 256)
    dummy_az = torch.randn(4, 2)

    # Train mode
    model.train()
    main_logits, az_logits = model(dummy_img, dummy_az, return_az_logits=True)
    assert main_logits.shape == (4, 2)
    assert az_logits.shape == (4, 4)

    # Eval mode
    model.eval()
    eval_logits = model(dummy_img, dummy_az)
    assert eval_logits.shape == (4, 2)

    # predict_proba
    probs = predict_proba(model, dummy_img, dummy_az)
    assert probs.shape == (4,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
