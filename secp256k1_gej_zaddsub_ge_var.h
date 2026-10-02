#include "secp256k1.h"
#include "group.h"
#include "field.h"
#include "util.h"

/**
 * ZADDSUB: Optimized Co-Z combined ADD and SUB (14M + 4S fast path) with Edge-Case Fallback.
 *
 * Input:
 *   Two Jacobian EC points P_a, P_s with a shared Z-coordinate
 *   (P_a.Z == P_s.Z), and an affine EC point S = (x, y).
 *   Precondition: none of P_a, P_s, S may be the point at infinity.
 *
 * Computes:
 *   res_a = P_a + S
 *   res_s = P_s - S
 *
 * Postcondition:
 *   res_a and res_s are returned fully normalized (X, Y, Z).
 *
 * Guaranteed Invariant:
 *   res_a and res_s share Z (res_a->z == res_s->z) whenever neither
 *   result is the point at infinity. Callers MUST check the infinity
 *   flag on both outputs before relying on shared Z.
 * 
 * Aliasing: res_a may alias pa, res_s may alias ps
 *   (in any combination). No other aliasing is supported.
 */
static void secp256k1_gej_zaddsub_ge_var(
    secp256k1_gej *res_a,
    secp256k1_gej *res_s,
    const secp256k1_gej *pa,
    const secp256k1_gej *ps,
    const secp256k1_ge *s
) {
    secp256k1_fe z_sq, z_cu, u, v;
    secp256k1_fe ha, hs, da, ds;
    secp256k1_fe hc, hc2, hc3;
    secp256k1_fe w, w1, w2, a;
    secp256k1_fe la, ls, la2, ls2;
    secp256k1_fe tmp, sum;

    /* Aliases to point components for readability */
    const secp256k1_fe *x  = &s->x;
    const secp256k1_fe *y  = &s->y;
    const secp256k1_fe *x1 = &pa->x;
    const secp256k1_fe *y1 = &pa->y;
    const secp256k1_fe *x2 = &ps->x;
    const secp256k1_fe *y2 = &ps->y;
    const secp256k1_fe *z  = &pa->z; /* pa and ps must share the same Z */

    /* ==========================================================
     * 1. SETUP PHASE 
     * ========================================================== */
    secp256k1_fe_sqr(&z_sq, z);              /* Z_sq = Z^2 */
    secp256k1_fe_mul(&z_cu, &z_sq, z);       /* Z_cu = Z^3 */

    secp256k1_fe_mul(&u, x, &z_sq);          /* U = x * Z^2 */
    secp256k1_fe_mul(&v, y, &z_cu);          /* V = y * Z^3 */

    /* H_a = U - X_1 */
    secp256k1_fe_negate(&ha, x1, 1);
    secp256k1_fe_add(&ha, &u);

    /* H_s = U - X_2 */
    secp256k1_fe_negate(&hs, x2, 1);
    secp256k1_fe_add(&hs, &u);

    /* D_a = V - Y_1 */
    secp256k1_fe_negate(&da, y1, 1);
    secp256k1_fe_add(&da, &v);

    /* D_s = -(V + Y_2) */
    sum = v; secp256k1_fe_add(&sum, y2);
    secp256k1_fe_negate(&ds, &sum, 2);

    /* ==========================================================
     * 2. HIGH-PERFORMANCE EDGE CASE FALLBACK
     * Triggered if P_a or P_s has the same X-coordinate as S.
     * ========================================================== */
    if (EXPECT(secp256k1_fe_normalizes_to_zero_var(&ha) ||
           secp256k1_fe_normalizes_to_zero_var(&hs), 0)) {
        
        /*
         * Each side is independently one of:
         *   H == 0, D == 0  -> doubling          (lambda = Y of that point)
         *   H == 0, D != 0  -> point at infinity
         *   H != 0          -> plain addition (lambda = H)
         *
         * After this block ha / hs hold lambda_a / lambda_s for every side
         * that is finite (double_var writes Y through rzr for a doubling, add_ge_var
         * writes H through rzr for a plain addition).
         */

        /* res_a = P_a + S */
        if (secp256k1_fe_normalizes_to_zero_var(&ha)) {
            if (secp256k1_fe_normalizes_to_zero_var(&da)) {
                secp256k1_gej_double_var(res_a, pa, &ha);         /* Pa = S, ha := Y1 */
            } else {
                secp256k1_gej_set_infinity(res_a);                /* Pa = -S */
            }
        } else {
            secp256k1_gej_add_ge_var(res_a, pa, s, &ha);          /* res_a->z = z*ha, ha := H */
        }

        /* res_s = P_s - S */
        if (secp256k1_fe_normalizes_to_zero_var(&hs)) {
            if (secp256k1_fe_normalizes_to_zero_var(&ds)) {
                secp256k1_gej_double_var(res_s, ps, &hs);         /* Ps = -S, hs := Y2 */
            } else {
                secp256k1_gej_set_infinity(res_s);                /* Ps = S */
            }
        } else {
            secp256k1_ge neg_s = *s;
            secp256k1_fe_negate(&neg_s.y, &neg_s.y, 1);
            secp256k1_fe_normalize_var(&neg_s.y);
            secp256k1_gej_add_ge_var(res_s, ps, &neg_s, &hs);    /* res_s->z = z*hs, hs := H */
        }
        
        if (!res_a->infinity && !res_s->infinity) {
            /* Both finite: rescale both onto z*ha*hs. */
            secp256k1_fe ha2, ha3, hs2, hs3, z_shared;

            secp256k1_fe_sqr(&ha2, &ha);
            secp256k1_fe_mul(&ha3, &ha2, &ha);
            secp256k1_fe_sqr(&hs2, &hs);
            secp256k1_fe_mul(&hs3, &hs2, &hs);

            secp256k1_fe_mul(&z_shared, &res_a->z, &hs);          /* z*ha*hs */
            secp256k1_fe_normalize_var(&z_shared);
            res_a->z = z_shared;
            res_s->z = z_shared;

            /* Scale res_a by hs */
            secp256k1_fe_mul(&res_a->x, &res_a->x, &hs2);
            secp256k1_fe_mul(&res_a->y, &res_a->y, &hs3);
            
            /* Scale res_s by ha */
            secp256k1_fe_mul(&res_s->x, &res_s->x, &ha2);
            secp256k1_fe_mul(&res_s->y, &res_s->y, &ha3);

            secp256k1_fe_normalize_var(&res_a->x);
            secp256k1_fe_normalize_var(&res_a->y);
            secp256k1_fe_normalize_var(&res_s->x);
            secp256k1_fe_normalize_var(&res_s->y);
        } else if (!res_a->infinity || !res_s->infinity) {
            /* Exactly one finite result */
            secp256k1_gej *fin = res_a->infinity ? res_s : res_a;

            secp256k1_fe_normalize_var(&fin->x);
            secp256k1_fe_normalize_var(&fin->y);
            secp256k1_fe_normalize_var(&fin->z);
        }

        return; 
    }

    /* ==========================================================
     * 3. FAST PATH: THE SHARED H_c CORE (14M + 4S)
     * ========================================================== */
    secp256k1_fe_mul(&hc, &ha, &hs);         /* H_c = H_a * H_s */
    secp256k1_fe_mul(&res_a->z, z, &hc);     /* Z_c = Z * H_c */
    secp256k1_fe_normalize_var(&res_a->z);
    res_s->z = res_a->z;                     

    secp256k1_fe_sqr(&hc2, &hc);             /* H_c2 = H_c^2 */
    secp256k1_fe_mul(&hc3, &hc2, &hc);       /* H_c3 = H_c^3 */

    /* Branch Variables */
    secp256k1_fe_mul(&w, &u, &hc2);          /* W   = U * H_c2 */
    secp256k1_fe_mul(&w1, x1, &hc2);         /* W_1 = X_1 * H_c2 */
    secp256k1_fe_mul(&w2, x2, &hc2);         /* W_2 = X_2 * H_c2 */

    secp256k1_fe_mul(&a, &v, &hc3);          /* A   = V * H_c3 */

    secp256k1_fe_mul(&la, &da, &hs);         /* L_a = D_a * H_s */
    secp256k1_fe_mul(&ls, &ds, &ha);         /* L_s = D_s * H_a */

    secp256k1_fe_sqr(&la2, &la);             /* L_a2 = L_a^2 */
    secp256k1_fe_sqr(&ls2, &ls);             /* L_s2 = L_s^2 */

    /* X_a = L_a2 - (W + W_1) */
    sum = w; secp256k1_fe_add(&sum, &w1);
    secp256k1_fe_negate(&tmp, &sum, 2);           
    res_a->x = la2;
    secp256k1_fe_add(&res_a->x, &tmp);
    secp256k1_fe_normalize_var(&res_a->x);

    /* X_s = L_s2 - (W + W_2) */
    sum = w; secp256k1_fe_add(&sum, &w2);
    secp256k1_fe_negate(&tmp, &sum, 2);
    res_s->x = ls2;
    secp256k1_fe_add(&res_s->x, &tmp);
    secp256k1_fe_normalize_var(&res_s->x);

    /* Y_a = L_a * (W - X_a) - A */
    secp256k1_fe_negate(&tmp, &res_a->x, 1);
    secp256k1_fe_add(&tmp, &w);             /* tmp = W - X_a */
    secp256k1_fe_mul(&res_a->y, &la, &tmp);
    secp256k1_fe_negate(&tmp, &a, 1);
    secp256k1_fe_add(&res_a->y, &tmp);

    /* Y_s = L_s * (W - X_s) + A */
    secp256k1_fe_negate(&tmp, &res_s->x, 1);
    secp256k1_fe_add(&tmp, &w);             /* tmp = W - X_s */
    secp256k1_fe_mul(&res_s->y, &ls, &tmp);
    secp256k1_fe_add(&res_s->y, &a);

    secp256k1_fe_normalize_var(&res_a->y);
    secp256k1_fe_normalize_var(&res_s->y);

    res_a->infinity = 0;
    res_s->infinity = 0;
}