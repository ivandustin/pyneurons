from jax.numpy import mean, square


def loss(model, x, y):
    yhat = model(x)
    return mean(square(y - yhat))
