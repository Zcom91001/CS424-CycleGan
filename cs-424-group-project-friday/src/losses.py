def mean(values):
    return sum(values) / max(1, len(values))


def l1(a, b):
    return mean([abs(x - y) for x, y in zip(a, b)])


def gan_loss(predictions, is_real, gan_mode="bce"):
    target = 1.0 if is_real else 0.0
    if gan_mode == "lsgan":
        return mean([(p - target) ** 2 for p in predictions])

    # bce-like surrogate to avoid log(0)
    eps = 1e-6
    clipped = [min(1.0 - eps, max(eps, p)) for p in predictions]
    if is_real:
        return mean([-1.0 * __import__("math").log(p) for p in clipped])
    return mean([-1.0 * __import__("math").log(1.0 - p) for p in clipped])
