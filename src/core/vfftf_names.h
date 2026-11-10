/* vfftf_names.h -- the external names of the FP32 unit (vfftf.c), included before vfft.c.
 *
 * The FP32 unit is vfft.c compiled again, so every symbol it defines with external linkage
 * gets an FP32 name here: the public plan functions become include/vfftf.h's, the plan
 * handle becomes vfftf_plan, and the engagement counters and diagnostics get an f.
 * vfft_malloc, vfft_free, vfft_error and vfft_measure_* are NOT defined by the FP32 unit:
 * they are the FP64 unit's and serve both (vfft.c guards their definitions). */
#ifndef VFFTF_NAMES_H
#define VFFTF_NAMES_H

/* the public plan functions and the handle (include/vfftf.h) */
#define vfft_create vfftf_create
#define vfft_execute vfftf_execute
#define vfft_destroy vfftf_destroy
#define vfft_info vfftf_info
#define vfft_plan vfftf_plan
#define vfft_plan_s vfftf_plan_s

/* the engagement counters (defined in vfft.c, declared extern by the engine headers) */
#define _vfft_create_race_count _vfftf_create_race_count
#define _vfft_il2d_col_mt_count _vfftf_il2d_col_mt_count
#define _vfft_il2d_row_mt_count _vfftf_il2d_row_mt_count
#define _vfft_ilfd_mt_count _vfftf_ilfd_mt_count
#define _vfft_ilfd_short_count _vfftf_ilfd_short_count
#define _vfft_ilnd_mt_count _vfftf_ilnd_mt_count
#define _vfft_ilpr_mt_count _vfftf_ilpr_mt_count
#define _vfft_pq_grp_count _vfftf_pq_grp_count
#define _vfft_pq_mt_count _vfftf_pq_mt_count
#define _vfft_scope_count _vfftf_scope_count
#define _vfft_tc_mt_dispatch_count _vfftf_tc_mt_dispatch_count
#define _vfft_zfsr_mt_count _vfftf_zfsr_mt_count
#define _vfft_zr2c_fold_mt_count _vfftf_zr2c_fold_mt_count
#define _vfft_zrb_mt_count _vfftf_zrb_mt_count
#define _vfft_zrf_mt_count _vfftf_zrf_mt_count
#define _vfft_ztt_mt_count _vfftf_ztt_mt_count
#define _vfft_zttr_mt_count _vfftf_zttr_mt_count

/* the diagnostics and tool entry points (vfft_diagnostics.h) */
#define vfft__ztt_mt_probe vfftf__ztt_mt_probe
#define vfft__ztt_mt_race_probe vfftf__ztt_mt_race_probe
#define vfft_il2d_col_mt_passes vfftf_il2d_col_mt_passes
#define vfft_il2d_row_mt_passes vfftf_il2d_row_mt_passes
#define vfft_ilfd_mt_passes vfftf_ilfd_mt_passes
#define vfft_ilfd_race_short_samples vfftf_ilfd_race_short_samples
#define vfft_ilnd_mt_passes vfftf_ilnd_mt_passes
#define vfft_ilpr_mt_passes vfftf_ilpr_mt_passes
#define vfft_isa vfftf_isa
#define vfft_plan_tc_workers vfftf_plan_tc_workers
#define vfft_pool_threads vfftf_pool_threads
#define vfft_pq_grp_passes vfftf_pq_grp_passes
#define vfft_pq_mt_passes vfftf_pq_mt_passes
#define vfft_tc_mt_dispatches vfftf_tc_mt_dispatches
#define vfft_wisdom_build vfftf_wisdom_build
#define vfft_wisdom_folder vfftf_wisdom_folder
#define vfft_wisdom_identity vfftf_wisdom_identity
#define vfft_wisdom_report vfftf_wisdom_report
#define vfft_wisdom_reread vfftf_wisdom_reread
#define vfft_zfsr_mt_passes vfftf_zfsr_mt_passes
#define vfft_zr2c_fold_mt_passes vfftf_zr2c_fold_mt_passes
#define vfft_zrb_mt_passes vfftf_zrb_mt_passes
#define vfft_zrf_mt_passes vfftf_zrf_mt_passes
#define vfft_ztt_mt_passes vfftf_ztt_mt_passes
#define vfft_zttr_mt_passes vfftf_zttr_mt_passes
#define vfft_tool_ilfd_chain vfftf_tool_ilfd_chain
#define vfft_tool_ilfd_destroy vfftf_tool_ilfd_destroy
#define vfft_tool_ilfd_exec vfftf_tool_ilfd_exec
#define vfft_tool_ilfd_exec_mt vfftf_tool_ilfd_exec_mt
#define vfft_tool_ilfd_race_inner vfftf_tool_ilfd_race_inner
#define vfft_tool_ilfd_rd_desc vfftf_tool_ilfd_rd_desc

#endif /* VFFTF_NAMES_H */
