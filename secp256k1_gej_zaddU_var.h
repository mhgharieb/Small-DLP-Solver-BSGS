#include "secp256k1.h"
#include "group.h"
#include "field.h"
#include "util.h"

/**
 * ZADDU: Optimized Co-Z addition with update: ZADDU (5M + 2S fast path) with Edge-Case Fallback.
 *
 * Input:
 *   Two Jacobian EC points P_a, S sharing one Z-coordinate (pa->z == s->z).
 *   Preconditions:
 *     - neither pa nor s is the point at infinity;
 *     - all input coordinates are normalized (magnitude 1).
 *
 * Computes:
 *   res_a = P_a + S
 *   s_out = S rescaled onto res_a's Z:
 *           s_out = (h_sq * S.X : h_sq*lambda * S.Y : Zc)
 *   h_sq  = lambda^2, where lambda is the Z-scaling factor (Zc = Z * lambda):
 *     - generic case (Pa.X != S.X): lambda = H_a = S.X - Pa.X
 *     - Pa == S  (res_a is a doubling) : lambda = Pa.Y  (Z3 = Y1*Z1 for a double)
 *     - Pa == -S (res_a is infinity)   : lambda = 1 (no scaling; s_out = S unchanged)
 *
 * Postcondition:
 *   res_a, s_out and h_sq are ALWAYS written and returned fully normalized.
 *   Callers must check res_a->infinity before relying on res_a->z == s_out->z.
 *
 * Aliasing: res_a may alias pa, s_out may alias s (in any combination).
 *   No other aliasing is supported.
 */
static void secp256k1_gej_zaddU_var(
    secp256k1_gej *res_a,
    secp256k1_gej *s_out,
    secp256k1_fe *h_sq,
    const secp256k1_gej *pa,
    const secp256k1_gej *s
) {
    secp256k1_fe u, v;
    secp256k1_fe ha, da, da2;
    secp256k1_fe ha2, w, w1, a;
    secp256k1_fe tmp, sum;

    const secp256k1_fe *x1 = &pa->x;
    const secp256k1_fe *y1 = &pa->y;
    const secp256k1_fe *z  = &pa->z;

    /* ==========================================================
     * 1. SETUP PHASE
     * ========================================================== */
    u = s->x;
    v = s->y;

    /* H_a = U - X_1 */
    secp256k1_fe_negate(&ha, x1, 1);
    secp256k1_fe_add(&ha, &u);

    /* D_a = V - Y_1 */
    secp256k1_fe_negate(&da, y1, 1);
    secp256k1_fe_add(&da, &v);

    /* ==========================================================
     * 2. HIGH-PERFORMANCE EDGE CASE FALLBACK
     * Triggered if P_a has the same X-coordinate as S.
     * ========================================================== */
    if (EXPECT(secp256k1_fe_normalizes_to_zero_var(&ha), 0)) {
        if (secp256k1_fe_normalizes_to_zero_var(&da)) {
            /* P_a == S: res_a = 2*P_a. double_var's rzr gives lambda = Y_1
             * directly (Z3 = Y1*Z1 for a doubling), captured before double()
             * runs internally, so this is safe even if res_a aliases pa. */
            secp256k1_fe ha3;

            secp256k1_gej_double_var(res_a, pa, &ha);

            secp256k1_fe_sqr(&ha2, &ha);
            secp256k1_fe_mul(&ha3, &ha2, &ha);
            /* Scale s_out by ha */
            secp256k1_fe_mul(&s_out->x, &u, &ha2);
            secp256k1_fe_mul(&s_out->y, &v, &ha3);

            secp256k1_fe_normalize_var(&res_a->x);
            secp256k1_fe_normalize_var(&res_a->y);
            secp256k1_fe_normalize_var(&res_a->z);
            secp256k1_fe_normalize_var(&s_out->x);
            secp256k1_fe_normalize_var(&s_out->y);
            s_out->z = res_a->z;
            s_out->infinity = 0;

            *h_sq = ha2;
            secp256k1_fe_normalize_var(h_sq);
        } else {
            /* P_a == -S: res_a = infinity. No valid scaling factor -- s_out is
             * returned unchanged (s is never aliased with res_a, only with
             * s_out, so it is still intact here), and h_sq = 1 as a no-op
             * placeholder. */
            secp256k1_gej_set_infinity(res_a);
            *s_out = *s;
            secp256k1_fe_set_int(h_sq, 1);
        }
        return;
    }

    /* ==========================================================
     * 3. FAST PATH (5M + 2S)
     * ========================================================== */
    secp256k1_fe_mul(&res_a->z, z, &ha);     /* Z_c = Z * H_a */
    secp256k1_fe_normalize_var(&res_a->z);
    s_out->z = res_a->z;

    secp256k1_fe_sqr(&ha2, &ha);             /* H_a^2 */
    
    secp256k1_fe_mul(&w, &u, &ha2);          /* W  = U  * H_a^2 */
    secp256k1_fe_mul(&w1, x1, &ha2);         /* W1 = X_1 * H_a^2 */

    /* A = V * H_a^3, using H_a^3 = W - W_1 (free: H_a*H_a^2 = (U-X1)*H_a^2) */
    sum = w;
    secp256k1_fe_negate(&tmp, &w1, 1);
    secp256k1_fe_add(&sum, &tmp);            /* sum = W - W1, mag 3 */
    secp256k1_fe_mul(&a, &v, &sum);

    secp256k1_fe_sqr(&da2, &da);             /* D_a^2 */

    /* X_a = D_a^2 - (W + W_1) */
    sum = w;
    secp256k1_fe_add(&sum, &w1);
    secp256k1_fe_negate(&tmp, &sum, 2);
    res_a->x = da2;
    secp256k1_fe_add(&res_a->x, &tmp);
    secp256k1_fe_normalize_var(&res_a->x);

    /* Y_a = D_a * (W - X_a) - A */
    secp256k1_fe_negate(&tmp, &res_a->x, 1);
    secp256k1_fe_add(&tmp, &w);
    secp256k1_fe_mul(&res_a->y, &da, &tmp);
    secp256k1_fe_negate(&tmp, &a, 1);
    secp256k1_fe_add(&res_a->y, &tmp);
    secp256k1_fe_normalize_var(&res_a->y);

    /* s_out = (W, A, Zc) */
    s_out->x = w;
    s_out->y = a;
    secp256k1_fe_normalize_var(&s_out->x);
    secp256k1_fe_normalize_var(&s_out->y);

    *h_sq = ha2;
    secp256k1_fe_normalize_var(h_sq);

    res_a->infinity = 0;
    s_out->infinity = 0;
}
