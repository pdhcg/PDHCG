# TODO

- [ ] **Distributed model transport:** Partition on rank 0 and send each rank
  only its local data and required metadata, instead of broadcasting the full
  problem before partitioning. Measure peak host memory and distribution time.

- [ ] **Processed problem representation:** Reduce `processed_qp_problem_t` to
  derived quadratic data (type, diagonal and low-rank middle data). Pass
  `qp_problem_t` separately for the original arrays and make ownership explicit.
