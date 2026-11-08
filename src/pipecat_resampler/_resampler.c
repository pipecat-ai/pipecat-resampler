/*
 * Copyright (c) 2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/* Python bindings for the resamplers, on the stable ABI (Py_LIMITED_API is
 * set by the build). */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

#include "_kernel.h"

/* What both resamplers hold. Objects are built whole in tp_new, so a call
 * that has released the GIL never sees them change. */
typedef struct {
    PyObject_HEAD
    filter_t *f;
    int in_rate;
    int out_rate;
    int channels;
} ResamplerObject;

typedef struct {
    ResamplerObject base;
    resampler_t *r;
} StreamResamplerObject;

static ResamplerObject *as_resampler(PyObject *self) { return (ResamplerObject *)self; }

static resampler_t *as_stream(PyObject *self) { return ((StreamResamplerObject *)self)->r; }

static void set_error(resampler_status_t status, const ResamplerObject *obj) {
    if (status == RESAMPLER_NO_MEMORY)
        PyErr_NoMemory();
    else if (status == RESAMPLER_TOO_LARGE)
        PyErr_Format(PyExc_ValueError,
                     "cannot resample from %d Hz to %d Hz: the filter would be too large",
                     obj->in_rate, obj->out_rate);
    else if (status == RESAMPLER_BAD_CHANNELS)
        PyErr_Format(PyExc_ValueError, "channels must be positive, got %d", obj->channels);
    else
        PyErr_Format(PyExc_ValueError, "sample rates must be positive, got %d and %d",
                     obj->in_rate, obj->out_rate);
}

/* Filters shared by every resampler between the same two rates, each freed
 * when the last resampler using it goes. Python creates and frees resamplers
 * with the GIL held, so the list needs no lock. */
typedef struct shared {
    struct shared *next;
    int in_rate;
    int out_rate;
    Py_ssize_t users;
    filter_t *f;
} shared_t;

static shared_t *shared_filters;

/* Returns the filter between two rates, made on first use, or NULL with
 * status set. */
static filter_t *take_filter(int in_rate, int out_rate, resampler_status_t *status) {
    for (shared_t *s = shared_filters; s; s = s->next) {
        if (s->in_rate == in_rate && s->out_rate == out_rate) {
            s->users++;
            return s->f;
        }
    }
    shared_t *s = malloc(sizeof *s);
    if (!s) {
        *status = RESAMPLER_NO_MEMORY;
        return NULL;
    }
    s->f = filter_new(in_rate, out_rate, status);
    if (!s->f) {
        free(s);
        return NULL;
    }
    s->in_rate = in_rate;
    s->out_rate = out_rate;
    s->users = 1;
    s->next = shared_filters;
    shared_filters = s;
    return s->f;
}

static void release_filter(filter_t *f) {
    for (shared_t **link = &shared_filters; *link; link = &(*link)->next) {
        shared_t *s = *link;
        if (s->f != f)
            continue;
        if (--s->users == 0) {
            *link = s->next;
            filter_free(s->f);
            free(s);
        }
        return;
    }
}

/* Allocates an object of type and takes its filter for (in_rate, out_rate,
 * channels=1), or returns NULL with an exception set. */
static ResamplerObject *create(PyTypeObject *type, PyObject *args, PyObject *kwds,
                               const char *format) {
    static char *keywords[] = {"in_rate", "out_rate", "channels", NULL};
    int in_rate, out_rate, channels = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, format, keywords, &in_rate, &out_rate,
                                     &channels))
        return NULL;
    allocfunc alloc = (allocfunc)PyType_GetSlot(type, Py_tp_alloc);
    ResamplerObject *obj = (ResamplerObject *)alloc(type, 0);
    if (!obj)
        return NULL;
    obj->in_rate = in_rate;
    obj->out_rate = out_rate;
    obj->channels = channels;
    resampler_status_t status = channels > 0 ? RESAMPLER_OK : RESAMPLER_BAD_CHANNELS;
    if (status == RESAMPLER_OK)
        obj->f = take_filter(in_rate, out_rate, &status);
    if (status != RESAMPLER_OK) {
        set_error(status, obj);
        Py_DECREF(obj);
        return NULL;
    }
    return obj;
}

static void free_object(PyObject *self) {
    PyTypeObject *type = Py_TYPE(self);
    if (as_resampler(self)->f)
        release_filter(as_resampler(self)->f);
    freefunc tp_free = (freefunc)PyType_GetSlot(type, Py_tp_free);
    tp_free(self);
    Py_DECREF(type);
}

/* Number of frames in view, or -1 with ValueError set if it holds a partial
 * frame. */
static Py_ssize_t whole_frames(const ResamplerObject *obj, const Py_buffer *view) {
    Py_ssize_t frame = (Py_ssize_t)(obj->channels * sizeof(int16_t));
    if (view->len % frame) {
        PyErr_Format(PyExc_ValueError,
                     "audio must be whole frames of 16-bit samples for %d channel(s), got %zd "
                     "bytes",
                     obj->channels, view->len);
        return -1;
    }
    return view->len / frame;
}

/* A new bytes object for count frames. */
static PyObject *frames_bytes(const ResamplerObject *obj, size_t count) {
    size_t size = count * (size_t)obj->channels * sizeof(int16_t);
    return PyBytes_FromStringAndSize(NULL, (Py_ssize_t)size);
}

static PyObject *get_in_rate(PyObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromLong(as_resampler(self)->in_rate);
}

static PyObject *get_out_rate(PyObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromLong(as_resampler(self)->out_rate);
}

static PyObject *get_channels(PyObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromLong(as_resampler(self)->channels);
}

static PyObject *get_kernel(PyObject *self, void *Py_UNUSED(closure)) {
    return PyUnicode_FromString(filter_kernel(as_resampler(self)->f));
}

/* ---- StreamResampler ---- */

static PyObject *StreamResampler_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    ResamplerObject *obj = create(type, args, kwds, "ii|i:StreamResampler");
    if (!obj)
        return NULL;
    resampler_status_t status;
    resampler_t *r = resampler_new(obj->f, obj->channels, &status);
    if (!r) {
        set_error(status, obj);
        Py_DECREF(obj);
        return NULL;
    }
    ((StreamResamplerObject *)obj)->r = r;
    return (PyObject *)obj;
}

static void StreamResampler_dealloc(PyObject *self) {
    resampler_free(as_stream(self));
    free_object(self);
}

/* Runs in_frames frames through the resampler into a new bytes object. */
static PyObject *run(PyObject *self, const void *in, size_t in_frames) {
    resampler_t *r = as_stream(self);
    PyObject *out = frames_bytes(as_resampler(self), resampler_output_count(r, in_frames));
    if (!out)
        return NULL;
    if (resampler_process(r, in, in_frames, (int16_t *)PyBytes_AsString(out)) != RESAMPLER_OK) {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }
    return out;
}

static PyObject *StreamResampler_process(PyObject *self, PyObject *audio) {
    Py_buffer view;
    if (PyObject_GetBuffer(audio, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    PyObject *out = NULL;
    Py_ssize_t frames = whole_frames(as_resampler(self), &view);
    if (frames >= 0)
        out = run(self, view.buf, (size_t)frames);
    PyBuffer_Release(&view);
    return out;
}

static PyObject *StreamResampler_flush(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    ResamplerObject *obj = as_resampler(self);
    resampler_t *r = as_stream(self);
    /* With nothing gone in since the last reset, the filter holds only
     * silence, so there's nothing to flush. */
    size_t delay = resampler_pending(r) ? filter_delay(obj->f) : 0;
    size_t samples = delay * (size_t)obj->channels;
    int16_t *zeros = calloc(samples ? samples : 1, sizeof(int16_t));
    if (!zeros)
        return PyErr_NoMemory();
    PyObject *out = run(self, zeros, delay);
    free(zeros);
    resampler_reset(r);
    return out;
}

static PyObject *StreamResampler_reset(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    resampler_reset(as_stream(self));
    Py_RETURN_NONE;
}

static PyObject *StreamResampler_get_delay(PyObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromSize_t(filter_delay(as_resampler(self)->f));
}

static PyMethodDef StreamResampler_methods[] = {
    {"process", StreamResampler_process, METH_O,
     "process(audio)\n--\n\n"
     "Resample a chunk of 16-bit PCM and return the resampled bytes.\n\n"
     "With more than one channel, the audio is interleaved frames. The result holds\n"
     "exactly the audio the input amounts to at the output rate, so equal chunks in\n"
     "give equal chunks out."},
    {"flush", StreamResampler_flush, METH_NOARGS,
     "flush()\n--\n\n"
     "Return the audio the filter is delaying and reset for the next stream.\n\n"
     "Empty when no audio has gone in since the resampler was created, flushed or reset."},
    {"reset", StreamResampler_reset, METH_NOARGS,
     "reset()\n--\n\n"
     "Discard the audio held in the filter."},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef StreamResampler_getset[] = {
    {"in_rate", get_in_rate, NULL, "Input sample rate in Hz.", NULL},
    {"out_rate", get_out_rate, NULL, "Output sample rate in Hz.", NULL},
    {"channels", get_channels, NULL, "Number of interleaved channels.", NULL},
    {"delay", StreamResampler_get_delay, NULL,
     "Input frames it takes for the last input to reach the output.", NULL},
    {"kernel", get_kernel, NULL,
     "Name of the dot-product kernel in use: avx2, sse2, neon or generic.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyType_Slot StreamResampler_slots[] = {
    {Py_tp_doc,
     "StreamResampler(in_rate, out_rate, channels=1)\n--\n\n"
     "Streaming resampler for 16-bit PCM whose output keeps pace with its input.\n\n"
     "A polyphase Kaiser-windowed sinc filter keeps the lower rate's band flat up to 90%\n"
     "of its Nyquist frequency and attenuates aliasing and images by about 90 dB. It\n"
     "delays the audio by a few milliseconds instead of holding output back. With more\n"
     "than one channel, audio is interleaved frames and each channel is filtered on its own."},
    {Py_tp_new, StreamResampler_new},
    {Py_tp_dealloc, StreamResampler_dealloc},
    {Py_tp_methods, StreamResampler_methods},
    {Py_tp_getset, StreamResampler_getset},
    {0, NULL},
};

static PyType_Spec StreamResampler_spec = {
    "pipecat_resampler._resampler.StreamResampler",
    sizeof(StreamResamplerObject),
    0,
    Py_TPFLAGS_DEFAULT,
    StreamResampler_slots,
};

/* ---- Resampler ---- */

static PyObject *Resampler_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return (PyObject *)create(type, args, kwds, "ii|i:Resampler");
}

static PyObject *Resampler_resample(PyObject *self, PyObject *audio) {
    ResamplerObject *obj = as_resampler(self);
    Py_buffer view;
    if (PyObject_GetBuffer(audio, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    PyObject *out = NULL;
    Py_ssize_t frames = whole_frames(obj, &view);
    if (frames >= 0)
        out = frames_bytes(obj, filter_resample_count(obj->f, (size_t)frames));
    if (out) {
        int16_t *dst = (int16_t *)PyBytes_AsString(out);
        resampler_status_t status;
        Py_BEGIN_ALLOW_THREADS
        status = filter_resample(obj->f, obj->channels, view.buf, (size_t)frames, dst);
        Py_END_ALLOW_THREADS
        if (status != RESAMPLER_OK) {
            Py_CLEAR(out);
            PyErr_NoMemory();
        }
    }
    PyBuffer_Release(&view);
    return out;
}

static PyMethodDef Resampler_methods[] = {
    {"resample", Resampler_resample, METH_O,
     "resample(audio)\n--\n\n"
     "Resample a whole clip of 16-bit PCM and return the resampled bytes.\n\n"
     "With more than one channel, the audio is interleaved frames. The clip is taken\n"
     "to be silent before and after, so nothing is held back for a later call."},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef Resampler_getset[] = {
    {"in_rate", get_in_rate, NULL, "Input sample rate in Hz.", NULL},
    {"out_rate", get_out_rate, NULL, "Output sample rate in Hz.", NULL},
    {"channels", get_channels, NULL, "Number of interleaved channels.", NULL},
    {"kernel", get_kernel, NULL,
     "Name of the dot-product kernel in use: avx2, sse2, neon or generic.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyType_Slot Resampler_slots[] = {
    {Py_tp_doc,
     "Resampler(in_rate, out_rate, channels=1)\n--\n\n"
     "One-shot resampler for whole clips of 16-bit PCM.\n\n"
     "It filters like StreamResampler. Its output lines up with its input and holds\n"
     "the clip's length at the output rate, rounded to the nearest sample. It releases\n"
     "the GIL while it works, and threads can share one. With more than one channel,\n"
     "audio is interleaved frames and each channel is filtered on its own."},
    {Py_tp_new, Resampler_new},
    {Py_tp_dealloc, free_object},
    {Py_tp_methods, Resampler_methods},
    {Py_tp_getset, Resampler_getset},
    {0, NULL},
};

static PyType_Spec Resampler_spec = {
    "pipecat_resampler._resampler.Resampler",
    sizeof(ResamplerObject),
    0,
    Py_TPFLAGS_DEFAULT,
    Resampler_slots,
};

/* ---- Module ---- */

static PyObject *filter_count(PyObject *Py_UNUSED(module), PyObject *Py_UNUSED(ignored)) {
    Py_ssize_t count = 0;
    for (shared_t *s = shared_filters; s; s = s->next)
        count++;
    return PyLong_FromSsize_t(count);
}

static PyMethodDef module_methods[] = {
    {"_filter_count", filter_count, METH_NOARGS,
     "_filter_count()\n--\n\n"
     "Number of filters the resamplers are sharing."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_resampler",
    "Polyphase sinc resamplers for 16-bit PCM, one-shot and streaming.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

static int add_type(PyObject *module, PyType_Spec *spec, const char *name) {
    PyObject *type = PyType_FromSpec(spec);
    if (!type)
        return -1;
    int result = PyModule_AddObjectRef(module, name, type);
    Py_DECREF(type);
    return result;
}

PyMODINIT_FUNC PyInit__resampler(void) {
    PyObject *module = PyModule_Create(&module_def);
    if (!module)
        return NULL;
    if (add_type(module, &Resampler_spec, "Resampler") < 0 ||
        add_type(module, &StreamResampler_spec, "StreamResampler") < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
