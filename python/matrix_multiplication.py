import tensorflow as tf
import time


def matrix_multiplication(matrices):
    """
    Multiplies a list of matrices in sequence.
    Args:
        matrices (list): List of tensors or arrays to multiply.
    Returns:
        tf.Tensor: The result of sequentially multiplying all matrices.
    """
    start_time = time.time()
    if not matrices or len(matrices) < 2:
        raise ValueError("Provide at least two matrices for multiplication.")
    result = matrices[0]
    for mat in matrices[1:]:
        result = tf.matmul(result, mat)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) * 1_000_000_000:.0f} nanoseconds")
    return result

if __name__ == "__main__":
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    c = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(matrix_multiplication([a, b, c]))
