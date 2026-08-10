function C = contact_closure_common()
%CONTACT_CLOSURE_COMMON Shared fail-closed utilities for the contact-closure chain.
%
% CONTRACT: this file is the single source of truth for the byte-exact
% hashing, canonical-path, filesystem-safety and numeric-closeness helpers
% that the contact-closure gate and study previously duplicated.  It also
% owns the STUDY EXECUTABLE SET and its harness root (see below).  Both the
% gate chain (contact_closure_gate + contact_gate_*) and the study chain
% (contact_closure_study + contact_study_*) obtain these helpers from here,
% so a mutation of any shared primitive is a single reviewable diff instead
% of a silent divergence between the two chains.
%
% RATIONALE (handle-struct pattern): this file returns one scalar struct of
% handles while each implementation lives in a separately named
% one-function module.  Callers write, e.g.:
%
%     common = contact_closure_common();
%     digest = common.file_sha256(path);
%
% The public struct keeps existing call sites compact; the separate modules
% make every primitive independently readable and auditable.
%
% HARNESS IDENTITY: the study report field harness_sha256 is no longer the
% hash of one monolithic file. It is the SHA-256 root of the complete STUDY
% EXECUTABLE SET: the entry/factory files, every one-function helper, shared
% utilities, solver-module inventory and provenance helpers. It is computed
% over the LF-joined, lexicographically sorted lines
%     <file name>:<file sha256>
% with no terminal LF -- the same grammar as the generator digest roots.
% The gate's report-binding revalidation and the independent Python checker
% (check_contact_closure_gate.py, _solver_execution_identity) recompute the
% identical root; all three sides must agree byte-for-byte.

C = struct();
C.file_sha256 = @contact_file_sha256;
C.file_bytes = @contact_file_bytes;
C.text_sha256 = @contact_text_sha256;
C.bytes_sha256 = @contact_bytes_sha256;
C.numeric_sha256 = @contact_numeric_sha256;
C.absolute_path = @contact_absolute_path;
C.is_same_or_child = @contact_is_same_or_child;
C.regular_nonsymlink = @contact_regular_nonsymlink;
C.regular_nonsymlink_directory = @contact_regular_nonsymlink_directory;
C.allclose = @contact_allclose;
C.logical_scalar = @contact_logical_scalar;
C.text_scalar = @contact_text_scalar;
C.utc_now = @contact_utc_now;
C.study_harness_files = @contact_study_harness_files;
C.study_harness_root = @contact_study_harness_root;
C.gate_module_files = @contact_gate_module_files;
C.gate_execution_root = @contact_gate_execution_root;
end
