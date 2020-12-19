import tensorflow as tf


def ffGE(t):
    piece = t / 0.710649
    shape = tf.square(1.0 + piece)
    GE = 1.0 / shape
    return GE


def ffGM(t):
    shape = ffGE(t)
    GM0 = 2.792847337
    GM = GM0 * shape
    return GM


def ffF2(t):
    f2 = (ffGM(t) - ffGE(t)) / (1 - (t / (4 * 0.938 * 0.938)))
    return f2


def ffF1(t):
    f1 = ffGM(t) - ffF2(t)
    return f1


def ffGA(t):
    ga = 1.2695
    ma = 1.026
    part = t / (ma * ma)
    dif = tf.square(1.0 - part)
    GA = ga / dif
    return GA


def galster(t):
    return ffF1(t), ffF2(t), ffGM(t)


a1_ep = tf.constant(-0.24, dtype=tf.float32)
a1_mp = tf.constant(0.12, dtype=tf.float32)
a1_mn = tf.constant(2.33, dtype=tf.float32)

b1_ep = tf.constant(10.98, dtype=tf.float32)
b1_mp = tf.constant(10.97, dtype=tf.float32)
b1_mn = tf.constant(14.72, dtype=tf.float32)

b2_ep = tf.constant(12.82, dtype=tf.float32)
b2_mp = tf.constant(18.86, dtype=tf.float32)
b2_mn = tf.constant(24.20, dtype=tf.float32)

b3_ep = tf.constant(0.12, dtype=tf.float32)
b3_mp = tf.constant(6.55, dtype=tf.float32)
b3_mn = tf.constant(84.1, dtype=tf.float32)

ea1ep = tf.constant(0.12, dtype=tf.float32)
ea1mp = tf.constant(0.04, dtype=tf.float32)
ea1mn = tf.constant(1.4, dtype=tf.float32)

eb1ep = tf.constant(0.19, dtype=tf.float32)
eb1mp = tf.constant(0.11, dtype=tf.float32)
eb1mn = tf.constant(1.7, dtype=tf.float32)

eb2ep = tf.constant(1.10, dtype=tf.float32)
eb2mp = tf.constant(0.28, dtype=tf.float32)
eb2mn = tf.constant(9.8, dtype=tf.float32)

eb3ep = tf.constant(6.8, dtype=tf.float32)
eb3mp = tf.constant(1.2, dtype=tf.float32)
eb3mn = tf.constant(41.0, dtype=tf.float32)

aa = tf.constant(1.7, dtype=tf.float32)
bb = tf.constant(3.3, dtype=tf.float32)
eaa = tf.constant(0.04, dtype=tf.float32)
ebb = tf.constant(0.32, dtype=tf.float32)

amp = tf.constant(0.9383, dtype=tf.float32)
amup = tf.constant(2.79, dtype=tf.float32)
amun = tf.constant(-1.91, dtype=tf.float32)


def kelly(t):
    tau = t / 4 / amp / amp

    gep = (1 + a1_ep * tau) / (
        1 + b1_ep * tau + b2_ep * tau * tau + b3_ep * tau * tau * tau
    )
    gmp = (
        amup
        * (1 + a1_mp * tau)
        / (1 + b1_mp * tau + b2_mp * tau * tau + b3_mp * tau * tau * tau)
    )
    gmn = (
        amun
        * (1 + a1_mn * tau)
        / (1 + b1_mn * tau + b2_mn * tau * tau + b3_mn * tau * tau * tau)
    )
    gen = aa * tau / (1 + tau * bb) / tf.math.square(1 + t / 0.71)

    e1 = tau / (1 + b1_ep * tau + b2_ep * tau * tau + b3_ep * tau * tau * tau)
    e2 = (
        (1 + a1_ep * tau)
        / tf.math.square(1 + b1_ep * tau + b2_ep * tau * tau + b3_ep * tau * tau * tau)
        * tau
    )
    e3 = (
        (1 + a1_ep * tau)
        / tf.math.square(1 + b1_ep * tau + b2_ep * tau * tau + b3_ep * tau * tau * tau)
        * tau
        * tau
    )
    e4 = (
        (1 + a1_ep * tau)
        / tf.math.square(1 + b1_ep * tau + b2_ep * tau * tau + b3_ep * tau * tau * tau)
        * tau
        * tau
        * tau
    )
    egep = tf.math.sqrt(
        tf.math.square(e1 * ea1ep)
        + tf.math.square(e2 * eb1ep)
        + tf.math.square(e3 * eb2ep)
        + tf.math.square(e4 * eb3ep)
    )

    e1m = tau / (1 + b1_mp * tau + b2_mp * tau * tau + b3_mp * tau * tau * tau)
    e2m = (
        (1 + a1_mp * tau)
        / tf.math.square(1 + b1_mp * tau + b2_mp * tau * tau + b3_mp * tau * tau * tau)
        * tau
    )
    e3m = (
        (1 + a1_mp * tau)
        / tf.math.square(1 + b1_mp * tau + b2_mp * tau * tau + b3_mp * tau * tau * tau)
        * tau
        * tau
    )
    e4m = (
        (1 + a1_mp * tau)
        / tf.math.square(1 + b1_mp * tau + b2_mp * tau * tau + b3_mp * tau * tau * tau)
        * tau
        * tau
        * tau
    )
    egmp = tf.math.sqrt(
        tf.math.square(e1m * ea1mp)
        + tf.math.square(e2m * eb1mp)
        + tf.math.square(e3m * eb2mp)
        + tf.math.square(e4m * eb3mp)
    )

    e1n = tau / (1 + b1_mn * tau + b2_mn * tau * tau + b3_mn * tau * tau * tau)
    e2n = (
        (1 + a1_mn * tau)
        / tf.math.square(1 + b1_mn * tau + b2_ep * tau * tau + b3_mn * tau * tau * tau)
        * tau
    )
    e3n = (
        (1 + a1_mn * tau)
        / tf.math.square(1 + b1_mn * tau + b2_ep * tau * tau + b3_mn * tau * tau * tau)
        * tau
        * tau
    )
    e4n = (
        (1 + a1_mn * tau)
        / tf.math.square(1 + b1_mn * tau + b2_ep * tau * tau + b3_mn * tau * tau * tau)
        * tau
        * tau
        * tau
    )
    egmn = tf.math.sqrt(
        tf.math.square(e1n * ea1mn)
        + tf.math.square(e2n * eb1mn)
        + tf.math.square(e3n * eb2mn)
        + tf.math.square(e4n * eb3mn)
    )

    e1t = tau / (1 + tau * bb) / tf.math.square(1 + t / 0.71)
    e2t = aa * tau * tau / tf.math.square(1 + tau * bb) / tf.math.square(1 + t / 0.71)
    egen = tf.math.sqrt(tf.math.square(e1t * eaa) + tf.math.square(e2t * ebb))

    f1p = (tau * gmp + gep) / (1 + tau)
    f2p = (gmp - gep) / (1 + tau)
    f1n = (tau * gmn + gen) / (1 + tau)
    f2n = (gmn - gen) / (1 + tau)

    gd = 1 / tf.math.square(1 + t / 0.71)

    ef1p = tf.math.sqrt(tf.math.square(tau * egmp) + tf.math.square(egep)) / (1 + tau)
    ef2p = tf.math.sqrt(tf.math.square(egmp) + tf.math.square(egep)) / (1 + tau)
    ef1n = tf.math.sqrt(tf.math.square(tau) + tf.math.square(egen)) / (1 + tau)
    ef2n = tf.math.sqrt(tf.math.square(egmn) + tf.math.square(egen)) / (1 + tau)

    ffF1 = f1p
    ffF2 = f2p
    ffGM = ffF1 + ffF2
    return ffF1, ffF2, ffGM
