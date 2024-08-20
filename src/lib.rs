pub use nalgebra::{OMatrix,Matrix,Dim,Dyn};
use nalgebra::linalg::SVD;
use nalgebra::partial_le;
use nalgebra::RealField;
use num_traits::identities::{One, Zero};

///trait so we can handle machine precision differences between f32 and f64
pub trait HasPrecision {
    fn epsilon() -> Self;
}

impl HasPrecision for f64 {
    fn epsilon() -> Self {
        f64::EPSILON
    }
}

impl HasPrecision for f32 {
    fn epsilon() -> Self {
        f32::EPSILON
    }
}

///Calculate the Moore-Penrose inverse of a real matrix using SVD
pub fn pinv<T:RealField+HasPrecision+One+Zero>(mat: OMatrix<T, Dyn, Dyn>) -> OMatrix<T, Dyn, Dyn>

{
    //first calculate our SVD decomposition
    let svd = SVD::new_unordered(mat, true, true);
    //now set all singular values with magnitude less than our machine precision to zero and invert
    //the others to create the pseudoinverse of sigma
    let sigma_t_diagonal = svd.singular_values.map(|entry: T::RealField| -> T::RealField {
        if partial_le(&entry.clone().abs(), &T::RealField::epsilon()) {
            T::RealField::zero()
        } else {
            T::RealField::one() / entry
        }
    });
    //Create our square matrix 'sigma' (singular values on the diagonal)
    let sigma_t = Matrix::from_diagonal(&sigma_t_diagonal);
    //our pseudoinverse is now svd.v_t.adjoint() * sigma_t * svd.u.adjoint
    svd.v_t.unwrap().adjoint() * sigma_t * svd.u.unwrap().adjoint()
}
