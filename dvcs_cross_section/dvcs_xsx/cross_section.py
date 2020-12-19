import math

import tensorflow as tf

from dvcs_xsx.four_vector import *
from dvcs_xsx.elastics import *
from dvcs_xsx import graphtrace


cos = tf.math.cos
sin = tf.math.sin
sqrt = tf.math.sqrt
T = tf.transpose
pi = tf.constant(math.pi, dtype=tf.float32)

# mass of the proton in GeV
M = tf.constant(0.93828, dtype=tf.float32)

# electromagnetic fine structure constant
alpha = tf.constant(0.0072973525693, dtype=tf.float32)


@graphtrace.trace_graph
def bh(xbj, t, Q2, k0, phi):
    depth_vector = tf.ones_like(xbj, dtype=tf.float32)

    ###################################
    ## Secondary Kinematic Variables ##
    ###################################

    # energy of the virtual photon
    nu = Q2 / (2.0 * M * xbj)

    # skewness parameter set by xbj, t, and Q^2
    xi = xbj * ((1.0 + (t / (2.0 * Q2))) / (2.0 - xbj + ((xbj * t) / Q2)))

    # gamma variable ratio of virtuality to energy of virtual photon
    gamma = sqrt(Q2) / nu

    # fractional energy of virtual photon
    y = sqrt(Q2) / (gamma * k0)

    # final lepton energy
    k0p = k0 * (1.0 - y)

    # minimum t value
    t0 = -(4.0 * xi * xi * M * M) / (1.0 - (xi * xi))

    # Lepton Angle Kinematics of initial lepton
    costl = -(1.0 / (sqrt(1.0 + gamma * gamma))) * (1.0 + (y * gamma * gamma / 2.0))
    sintl = (gamma / (sqrt(1.0 + gamma * gamma))) * sqrt(
        1.0 - y - (y * y * gamma * gamma / 4.0)
    )

    # Lepton Angle Kinematics of final lepton
    sintlp = sintl / (1.0 - y)
    costlp = (costl + y * sqrt(1.0 + gamma * gamma)) / (1.0 - y)

    # final proton energy
    p0p = M - (t / (2.0 * M))

    # ratio of longitudinal to transverse virtual photon flux
    eps = (1.0 - y - 0.25 * y * y * gamma * gamma) / (
        1.0 - y + 0.5 * y * y + 0.25 * y * y * gamma * gamma
    )

    # angular kinematics of outgoing photon
    cost = -(1 / (sqrt(1 + gamma * gamma))) * (
        1 + (0.5 * gamma * gamma) * ((1 + (t / Q2)) / (1 + ((xbj * t) / (Q2))))
    )
    cost = tf.math.maximum(cost, -1.0)
    sint = sqrt(1.0 - cost * cost)

    # outgoing photon energy
    q0p = (sqrt(Q2) / gamma) * (1 + ((xbj * t) / Q2))

    # conversion from GeV to NanoBarn
    jacobian = 2.0 * pi
    conversion = (0.1973 * 0.1973) * 10000000 * jacobian / 4

    # ratio of momentum transfer to proton mass
    tau = -t / (4.0 * M * M)

    ###############################################################################
    ## Creates arrays of 4-vector kinematics uses in Bethe Heitler Cross Section ##
    ###############################################################################

    # initial proton 4-momentum
    p = T(
        tf.convert_to_tensor(
            [
                M * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
            ]
        )
    )

    # initial lepton 4-momentum
    k = T(
        tf.convert_to_tensor(
            [
                k0 * depth_vector,
                k0 * sintl * depth_vector,
                0.0 * depth_vector,
                k0 * costl * depth_vector,
            ]
        )
    )

    # final lepton 4-momentum
    kp = T(
        tf.convert_to_tensor(
            [
                k0p * depth_vector,
                k0p * sintlp * depth_vector,
                0.0 * depth_vector,
                k0p * costlp * depth_vector,
            ]
        )
    )

    # virtual photon 4-momentum
    q = k - kp

    ##################################
    ## Creates four vector products ##
    ##################################
    plp = product(p, p)  # pp
    qq = product(q, q)  # qq
    kk = product(k, k)  # kk
    kkp = product(k, kp)  # kk'
    kq = product(k, q)  # kq
    pk = product(k, p)  # pk
    pkp = product(kp, p)  # pk'

    # sets the Mandelstam variables s which is the center of mass energy
    s = kk + (2 * pk) + plp

    # the Gamma factor in front of the cross section
    Gamma = (alpha ** 3) / (
        16.0 * (pi ** 2) * ((s - M * M) ** 2) * sqrt(1.0 + gamma ** 2) * xbj
    )

    phi = phi * 0.0174532951  # radian conversion

    # final real photon 4-momentum
    qp = T(
        tf.convert_to_tensor(
            [
                q0p * depth_vector,
                q0p * sint * T(cos(phi)),
                q0p * sint * T(sin(phi)),
                q0p * cost * depth_vector,
            ]
        )
    )

    # momentum transfer Δ from the initial proton to the final proton
    d = q - qp

    # final proton momentum
    pp = p + d

    # average initial proton momentum
    P = 0.5 * (p + pp)

    # 4-vector products of variables multiplied by spin vectors
    ppSL = ((M) / (sqrt(1.0 + gamma ** 2))) * (
        xbj * (1.0 - (t / Q2)) - (t / (2.0 * M ** 2))
    )
    kSL = (
        ((Q2) / (sqrt(1.0 + gamma ** 2)))
        * (1.0 + 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )
    kpSL = (
        ((Q2) / (sqrt(1 + gamma ** 2)))
        * (1 - y - 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )

    # 4-vector products denoted in the paper by the commented symbols
    kd = product(k, d)  # dΔ
    kpd = product(kp, d)  # k'Δ
    kP = product(k, P)  # kP
    kpP = product(kp, P)  # k'P
    kqp = product(k, qp)  # kq'
    kpqp = product(kp, qp)  # k'q'
    dd = product(d, d)  # ΔΔ
    Pq = product(P, q)  # Pq
    Pqp = product(P, qp)  # Pq'
    qd = product(q, d)  # qΔ
    qpd = product(qp, d)  # q'Δ

    # expresssions used that appear in coefficient calculations
    Dplus = (1 / (2 * kpqp)) - (1 / (2 * kqp))
    Dminus = (1 / (2 * kpqp)) + (1 / (2 * kqp))

    # calculates BH
    AUUBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (4.0 * tau * (kP * kP + kpP * kpP)) - ((tau + 1.0) * (kd * kd + kpd * kpd))
    )
    BUUBH = ((32.0 * M * M) / (kqp * kpqp)) * (kd * kd + kpd * kpd)

    # converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBH = (Gamma / t ** 2) * AUUBH * conversion * 2
    con_BUUBH = (Gamma / t ** 2) * BUUBH * conversion * 2

    ffF1, ffF2, ffGM = kelly(-t)

    # unpolarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # we use the Galster Form Factors as approximations
    bhAUU = con_AUUBH * ((ffF1 * ffF1) + (tau * ffF2 * ffF2))
    bhBUU = con_BUUBH * (tau * ffGM * ffGM)

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBH = bhAUU + bhBUU

    #    sigmas = T(tf.stack([XSXUU, XSXLU, XSXUL, XSXLL, XSXALU, XSXAUL, XSXALL]))
    #    gather_nd_idxs = tf.stack(
    #        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    #    )
    #    return tf.gather_nd(sigmas, gather_nd_idxs)

    return XSXUUBH


if __name__ == "__main__":
    # quick BH test
    xbj = tf.constant(0.343)
    t = tf.constant(-0.172)
    Q2 = tf.constant(1.82)
    k0 = tf.constant(5.75)
    phi = tf.constant(0.0)
    print(bh(xbj, t, Q2, k0, phi))
