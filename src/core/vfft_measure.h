/* vfft_measure.h -- the public measurement calls (vfft.h, "the measurement
 * scope"): vfft_measure_configure, vfft_measure_begin, vfft_measure_end,
 * vfft_measure_confine, vfft_measure_describe; and vfft_measure_scopes
 * (vfft_diagnostics.h).
 *
 * The scope itself -- the machine-wide lock, the pin to a P-core, the sibling
 * guard, the priority -- is common/support/race_scope.h, and vfft_create
 * enters it by itself around every race. These calls put the SAME scope
 * around code a caller times: a program that compares the library with
 * another one measures both under the conditions the library's own races ran
 * under, through the library, with no pin or guard code of its own.
 *
 * POSITION IN vfft.c: after race_scope.h and the pool (threads.h).
 */
#ifndef VFFT_MEASURE_H
#define VFFT_MEASURE_H

void vfft_measure_configure(const vfft_measure_config_t *config)
{
    _vfs_cfg_t c;
    memset(&c, 0, sizeof c);
    if (config)
    {
        c.pin = (config->pin == VFFT_MEASURE_PIN_OFF) ? 1 : (config->pin == VFFT_MEASURE_PIN_CORE) ? 2 : 0;
        c.pin_core = config->pin_core;
        c.guard = (config->guard == VFFT_MEASURE_GUARD_OFF) ? 1 : (config->guard == VFFT_MEASURE_GUARD_PAUSE) ? 2 : 0;
        c.priority = (config->priority == VFFT_MEASURE_PRIORITY_LEAVE) ? 1
                   : (config->priority == VFFT_MEASURE_PRIORITY_PROCESS) ? 2 : 0;
        c.lock_wait_ms = config->lock_wait_ms > 0 ? config->lock_wait_ms : 0;
    }
    _vfs_cfg = c;
}

int vfft_measure_begin(void)
{
    return _vfs_enter();
}

void vfft_measure_end(void)
{
    if (_vfs.auto_on && _vfs.depth == 1)
        return;   /* the scope a create entered is the create's to leave */
    _vfs_leave();
}

unsigned long long vfft_measure_confine(unsigned long long mask)
{
    if (mask == 0)
    {   /* the first logical CPU of each P-core */
        int k, c;
        for (k = 0; (c = vfft_topo_pcore_cpu(k)) >= 0; k++)
            if (c < 64) mask |= 1ull << c;
    }
    if (mask == 0)
        return 0;
#if defined(_WIN32)
    return SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)mask) ? mask : 0;
#elif defined(__linux__)
    {
        /* sched_setaffinity(0) sets the CALLING thread's mask; every thread
         * created after it inherits it */
        cpu_set_t s;
        int c;
        CPU_ZERO(&s);
        for (c = 0; c < 64; c++)
            if (mask & (1ull << c)) CPU_SET(c, &s);
        if (sched_setaffinity(0, sizeof s, &s) != 0)
            return 0;
        _vfs_proc_set = s;              /* the set this process may run on, from now */
        _vfs_proc_set_state = 1;
        return mask;
    }
#else
    return 0;
#endif
}

const char *vfft_measure_describe(char *buf, size_t n)
{
    static const char *const gk[] = { "none", "TPAUSE (C0.2)", "MONITORX/MWAITX", "PAUSE spinner" };
    char pin[64], guard[80];
    if (!buf || !n)
        return "";
    if (!_vfs_last.valid)
    {
        snprintf(buf, n, "no measurement scope has run on this thread");
        return buf;
    }
    if (_vfs_last.core >= 0)
        snprintf(pin, sizeof pin, "logical CPU %d (%s)", _vfs_last.core, _vfs_last.pcore ? "a P-core" : "NOT a P-core");
    else
        snprintf(pin, sizeof pin, "none");
    if (_vfs_last.guard_kind)
        snprintf(guard, sizeof guard, "logical CPU %d held by %s", _vfs_last.guard_cpu, gk[_vfs_last.guard_kind & 3]);
    else
        snprintf(guard, sizeof guard, "none");
    snprintf(buf, n, "pin: %s; sibling guard: %s; priority: %s; measurement lock: %s%s",
             pin, guard,
             _vfs_last.prio == 2 ? "process raised" : _vfs_last.prio == 1 ? "thread raised" : "unchanged",
             _vfs_last.lock_state > 0 ? "held" : _vfs_last.lock_state == 0 ? "NOT obtained" : "none on this system",
             _vfs_last.flags ? "; winners are not saved" : "");
    return buf;
}

long vfft_measure_scopes(void) { return _vfft_scope_count; }

#endif /* VFFT_MEASURE_H */
