/// Available GPU particle sorting algorithms.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum SortAlgorithm {
    Bitonic,
    Radix,
}
