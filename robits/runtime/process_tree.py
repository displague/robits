"""Own a subprocess tree even when its original parent exits first."""
import os
import subprocess


class ProcessTree:
    def __init__(self, command, env):
        self.job = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes as w
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
            kernel.CreateJobObjectW.restype = w.HANDLE
            kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
            kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
            kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
            kernel.CloseHandle.argtypes = [w.HANDLE]
            class Basic(ctypes.Structure):
                _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", w.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_processes", w.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
            class Extended(ctypes.Structure):
                _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]
            self.kernel = kernel
            self.job = kernel.CreateJobObjectW(None, None)
            if not self.job:
                raise ctypes.WinError(ctypes.get_last_error())
            info = Extended()
            info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                kernel.CloseHandle(self.job)
                self.job = None
                raise ctypes.WinError(ctypes.get_last_error())
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NO_WINDOW | 0x4) if os.name == "nt" else 0)
            if self.job:
                if not self.kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                    raise ctypes.WinError(ctypes.get_last_error())
                resume = ctypes.WinDLL("ntdll").NtResumeProcess
                resume.argtypes, resume.restype = [w.HANDLE], w.LONG
                if resume(int(self.process._handle)) < 0:
                    raise OSError("Could not resume owned MCP server process")
        except Exception:
            self.close()
            raise

    def close(self):
        process = getattr(self, "process", None)
        if self.job:
            # The job tracks descendants even after the original process exits.
            self.kernel.TerminateJobObject(self.job, 1)
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif process is not None and os.name != "nt":
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
