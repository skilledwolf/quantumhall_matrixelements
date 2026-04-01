# Conventions

## Magnetic-field sign

The package uses

$$
\sigma = \mathrm{sgn}(q B_z)
$$

as its public field-orientation convention. The default is
`sign_magneticfield = -1`, which corresponds to the package's electron in
positive `B_z` convention.

Passing `sign_magneticfield=+1` returns the opposite-field quantity with the
appropriate phase convention already applied. No manual conjugation or angle
flip is required.

## Momentum and magnetic-length units

The basic inputs are polar wavevector arrays `q_magnitudes` and `q_angles`.
They must have the same shape; the public APIs do not broadcast them.

All formulas depend on the dimensionless combination

$$
|q| \ell_B.
$$

This means:

- If your inputs are already dimensionless `|q| l_B`, pass them directly and leave `lB=1`.
- If your inputs are physical wavevectors in inverse-length units, pass those values together with the desired `lB`.

The same rule applies to the symmetric-gauge form-factor helpers.

## Landau-gauge form factor

The implemented plane-wave matrix element is

$$
F_{n',n}^{\sigma}(\mathbf{q})
=
i^{|n-n'|}
e^{i \sigma (n' - n)\theta_{\mathbf{q}}}
\sqrt{\frac{n_{<}!}{n_{>}!}}
\left( \frac{| \mathbf{q} | \ell_B}{\sqrt{2}} \right)^{|n-n'|}
L_{n_<}^{|n-n'|}\!\left( \frac{| \mathbf{q} |^2 \ell_B^2}{2} \right)
e^{-| \mathbf{q} |^2 \ell_B^2 / 4},
$$

where `n_< = min(n, n')` and `n_> = max(n, n')`.

The underlying Landau-gauge wavefunction convention is

$$
\Psi_{nX}^{\sigma}(x, y)
=
\frac{e^{i \sigma X y \ell_B^{-2}}}{\sqrt{L_y}} i^n \phi_n(x - X),
\qquad
X = \sigma k_y \ell_B^2.
$$

## Exchange kernel

The exchange kernel is

$$
X_{n_1 m_1 n_2 m_2}^{\sigma}(\mathbf{G})
=
\int \frac{d^2 q}{(2\pi)^2}
V(q)
F_{m_1,n_1}^{\sigma}(\mathbf{q})
F_{n_2,m_2}^{\sigma}(-\mathbf{q})
e^{i \sigma (\mathbf{q} \times \mathbf{G})_z \ell_B^2}.
$$

For most calculations, `method="laguerre"` is the right backend. The full
materialized exchange tensor scales like `O(nmax^4)` per `G`, so large jobs
usually belong on the compressed or Fock-constructor APIs.

The explicit tensor order is `(G, n1, m1, n2, m2)`. In this Landau-gauge
notation, `m1` and `m2` are still Landau-level labels, not symmetric-gauge
guiding-center orbitals.

## Symmetric gauge

In the symmetric-gauge basis `|n, m>`, the density operator factorizes as

$$
\langle n', m' | e^{i \mathbf{q} \cdot \mathbf{r}} | n, m \rangle
=
F_{n',n}^{\sigma}(\mathbf{q}) G_{m',m}^{-\sigma}(\mathbf{q}).
$$

The guiding-center sector therefore has the opposite chirality to the
cyclotron sector. The helper `get_guiding_center_form_factors(...)` handles
that sign flip internally.

## Coulomb scaling

For the built-in Coulomb interaction, the package assumes

$$
V(q) = \kappa \frac{2 \pi}{q},
$$

with `q` expressed in `1 / l_B` units. With `kappa=1`, the resulting exchange
quantities are in the Coulomb energy scale `e^2 / (\epsilon l_B)`. If you pass
a callable `potential(q)`, its return value sets the overall energy units.
