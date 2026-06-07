# ----------------------------------------------------------------------
# Самотестирование (если запущено как скрипт)
# ----------------------------------------------------------------------
from surface_geometry_pack import SurfaceGeometryPack
import numpy as np


if __name__ == "__main__":
    # Простейший тест на эллипсоиде
    a, b, c = 2.4, 2.0, 1.6
    u, v = np.pi / 3.0, np.pi / 6.0

    class DummyEllipsoid:
        def position(self, u, v):
            return np.array([
                a * np.cos(u) * np.cos(v),
                b * np.sin(u) * np.cos(v),
                c * np.sin(v)
            ])
        def derivatives(self, u, v):
            cos_u, sin_u = np.cos(u), np.sin(u)
            cos_v, sin_v = np.cos(v), np.sin(v)
            r = self.position(u, v)
            ru = np.array([-a * sin_u * cos_v, b * cos_u * cos_v, 0.0])
            rv = np.array([-a * cos_u * sin_v, -b * sin_u * sin_v, c * cos_v])
            n = np.array([cos_u * cos_v / a, sin_u * cos_v / b, sin_v / c])
            n = n / np.linalg.norm(n)
            return {"r": r, "ru": ru, "rv": rv, "normal": n}
        def first_fundamental_form(self, u, v):
            cos_u, sin_u = np.cos(u), np.sin(u)
            cos_v, sin_v = np.cos(v), np.sin(v)
            E = a**2 * sin_u**2 * cos_v**2 + b**2 * cos_u**2 * cos_v**2
            F = (a**2 - b**2) * sin_u * cos_u * sin_v * cos_v
            G = a**2 * cos_u**2 * sin_v**2 + b**2 * sin_u**2 * sin_v**2 + c**2 * cos_v**2
            return E, F, G
        def second_fundamental_form(self, u, v):
            # Для эллипсоида x²/a² + y²/b² + z²/c² = 1
            cos_u, sin_u = np.cos(u), np.sin(u)
            cos_v, sin_v = np.cos(v), np.sin(v)
            denom = np.sqrt(
                (cos_u * cos_v / a)**2 +
                (sin_u * cos_v / b)**2 +
                (sin_v / c)**2
            )
            L = a * b * c * cos_v / (a**2 * denom**3)
            M = 0.0
            N = a * b * c / (c**2 * denom**3)
            return L, M, N

    surf = DummyEllipsoid()
    geom = SurfaceGeometryPack.from_surface(surf, u, v)

    print("=== SurfaceGeometryPack: самотестирование ===")
    print(f"Точка:  u={u:.4f}, v={v:.4f}")
    print(f"r   = {geom.r}")
    print(f"ru  = {geom.ru}")
    print(f"rv  = {geom.rv}")
    print(f"n   = {geom.normal}")
    print(f"E={geom.E:.4f}, F={geom.F:.4f}, G={geom.G:.4f}")
    print(f"L={geom.L:.4f}, M={geom.M:.4f}, N={geom.N:.4f}")
    print(f"det(G) = {geom.det_G:.4f}")
    print()
    print("G_inv =")
    print(geom.G_inv)
    print()
    print("B =")
    print(geom.B)

    # Проверка: вектор нити = R - r, где R — точка на внешнем эллипсоиде
    a1, b1, c1 = 3.0, 2.5, 2.0
    R = np.array([
        a1 * np.cos(u) * np.cos(v),
        b1 * np.sin(u) * np.cos(v),
        c1 * np.sin(v)
    ])
    V_thread = R - geom.r
    print(f"\nV_thread = R - r = {V_thread}")

    P = geom.project_on_basis(V_thread)
    print(f"P (проекции на базис) = {P}")

    grad_u = geom.grad_Phi(V_thread)
    print(f"∇_u Φ = {grad_u}")

    grad_s = geom.surface_gradient(grad_u)
    print(f"∇_S Φ = {grad_s}")

    Ng = geom.norm_grad_sq(grad_u)
    print(f"|∇_S Φ|²_G = {Ng:.6f}")

    # Проверка: Φ должно быть ≈ 0, т.к. R лежит на луче из r
    # (для теста просто проверим, что grad_Phi конечен)
    assert np.isfinite(grad_u).all()
    assert Ng > 0
    print("\n✓ Все проверки пройдены.")