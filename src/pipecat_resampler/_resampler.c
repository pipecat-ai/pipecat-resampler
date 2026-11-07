/*
 * Copyright (c) 2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/* Python bindings for the resampler, on the stable ABI (Py_LIMITED_API is
 * set by the build). */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

#include "_kernel.h"

typedef struct {
    PyObject_HEAD
    resampler_t *r;
    int in_rate;
    int out_rate;
} ResamplerObject;

static resampler_t *get(PyObject *self) {
    resampler_t *r = ((ResamplerObject *)self)->r;
    if (!r)
        PyErr_SetString(PyExc_RuntimeError, "Resampler.__init__() was not called");
    return r;
}

static int Resampler_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *keywords[] = {"in_rate", "out_rate", "channels", NULL};
    int in_rate, out_rate, channels = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|i:Resampler", keywords, &in_rate, &out_rate,
                                     &channels))
        return -1;
    resampler_status_t status;
    resampler_t *r = resampler_new(in_rate, out_rate, channels, &status);
    if (!r) {
        if (status == RESAMPLER_NO_MEMORY)
            PyErr_NoMemory();
        else if (status == RESAMPLER_TOO_LARGE)
            PyErr_Format(PyExc_ValueError,
                         "cannot resample from %d Hz to %d Hz: the filter would be too large",
                         in_rate, out_rate);
        else if (status == RESAMPLER_BAD_CHANNELS)
            PyErr_Format(PyExc_ValueError, "channels must be positive, got %d", channels);
        else
            PyErr_Format(PyExc_ValueError, "sample rates must be positive, got %d and %d",
                         in_rate, out_rate);
        return -1;
    }
    ResamplerObject *obj = (ResamplerObject *)self;
    resampler_free(obj->r);
    obj->r = r;
    obj->in_rate = in_rate;
    obj->out_rate = out_rate;
    return 0;
}

static void Resampler_dealloc(PyObject *self) {
    PyTypeObject *type = Py_TYPE(self);
    resampler_free(((ResamplerObject *)self)->r);
    freefunc tp_free = (freefunc)PyType_GetSlot(type, Py_tp_free);
    tp_free(self);
    Py_DECREF(type);
}

/* Runs in_frames frames through the resampler into a new bytes object. */
static PyObject *run(resampler_t *r, const void *in, size_t in_frames) {
    size_t count = resampler_output_count(r, in_frames);
    size_t size = count * (size_t)resampler_channels(r) * sizeof(int16_t);
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)size);
    if (!out)
        return NULL;
    if (resampler_process(r, in, in_frames, (int16_t *)PyBytes_AsString(out)) != RESAMPLER_OK) {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }
    return out;
}

static PyObject *Resampler_process(PyObject *self, PyObject *audio) {
    resampler_t *r = get(self);
    if (!r)
        return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(audio, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    PyObject *out = NULL;
    int channels = resampler_channels(r);
    Py_ssize_t frame = (Py_ssize_t)(channels * sizeof(int16_t));
    if (view.len % frame)
        PyErr_Format(PyExc_ValueError,
                     "audio must be whole frames of 16-bit samples for %d channel(s), got %zd "
                     "bytes",
                     channels, view.len);
    else
        out = run(r, view.buf, (size_t)(view.len / frame));
    PyBuffer_Release(&view);
    return out;
}

static PyObject *Resampler_flush(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    resampler_t *r = get(self);
    if (!r)
        return NULL;
    /* With nothing gone in since the last reset, the filter holds only
     * silence, so there's nothing to flush. */
    size_t delay = resampler_pending(r) ? resampler_delay(r) : 0;
    size_t samples = delay * (size_t)resampler_channels(r);
    int16_t *zeros = calloc(samples ? samples : 1, sizeof(int16_t));
    if (!zeros)
        return PyErr_NoMemory();
    PyObject *out = run(r, zeros, delay);
    free(zeros);
    resampler_reset(r);
    return out;
}

static PyObject *Resampler_reset(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    resampler_t *r = get(self);
    if (!r)
        return NULL;
    resampler_reset(r);
    Py_RETURN_NONE;
}

static PyObject *Resampler_get_in_rate(PyObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromLong(((ResamplerObject *)self)->in_rate);
}

static PyObject *Resampler_get_out_rate(PyObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromLong(((ResamplerObject *)self)->out_rate);
}

static PyObject *Resampler_get_channels(PyObject *self, void *Py_UNUSED(closure)) {
    resampler_t *r = get(self);
    return r ? PyLong_FromLong(resampler_channels(r)) : NULL;
}

static PyObject *Resampler_get_delay(PyObject *self, void *Py_UNUSED(closure)) {
    resampler_t *r = get(self);
    return r ? PyLong_FromSize_t(resampler_delay(r)) : NULL;
}

static PyObject *Resampler_get_kernel(PyObject *self, void *Py_UNUSED(closure)) {
    resampler_t *r = get(self);
    return r ? PyUnicode_FromString(resampler_kernel(r)) : NULL;
}

static PyMethodDef Resampler_methods[] = {
    {"process", Resampler_process, METH_O,
     "process(audio)\n--\n\n"
     "Resample a chunk of 16-bit PCM and return the resampled bytes.\n\n"
     "With more than one channel, the audio is interleaved frames. The result holds\n"
     "exactly the audio the input amounts to at the output rate, so equal chunks in\n"
     "give equal chunks out."},
    {"flush", Resampler_flush, METH_NOARGS,
     "flush()\n--\n\n"
     "Return the audio the filter is delaying and reset for the next stream.\n\n"
     "Empty when no audio has gone in since the resampler was created, flushed or reset."},
    {"reset", Resampler_reset, METH_NOARGS,
     "reset()\n--\n\n"
     "Discard the audio held in the filter."},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef Resampler_getset[] = {
    {"in_rate", Resampler_get_in_rate, NULL, "Input sample rate in Hz.", NULL},
    {"out_rate", Resampler_get_out_rate, NULL, "Output sample rate in Hz.", NULL},
    {"channels", Resampler_get_channels, NULL, "Number of interleaved channels.", NULL},
    {"delay", Resampler_get_delay, NULL,
     "Input frames it takes for the last input to reach the output.", NULL},
    {"kernel", Resampler_get_kernel, NULL,
     "Name of the dot-product kernel in use: avx2, sse2, neon or generic.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyType_Slot Resampler_slots[] = {
    {Py_tp_doc,
     "Resampler(in_rate, out_rate, channels=1)\n--\n\n"
     "Streaming resampler for 16-bit PCM whose output keeps pace with its input.\n\n"
     "A polyphase Kaiser-windowed sinc filter keeps the lower rate's band flat up to 90%\n"
     "of its Nyquist frequency and attenuates aliasing and images by about 90 dB. It\n"
     "delays the audio by a few milliseconds instead of holding output back. With more\n"
     "than one channel, audio is interleaved frames and each channel is filtered on its own."},
    {Py_tp_new, PyType_GenericNew},
    {Py_tp_init, Resampler_init},
    {Py_tp_dealloc, Resampler_dealloc},
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

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_resampler",
    "Streaming audio resampler whose output keeps pace with its input.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__resampler(void) {
    PyObject *module = PyModule_Create(&module_def);
    if (!module)
        return NULL;
    PyObject *type = PyType_FromSpec(&Resampler_spec);
    if (!type || PyModule_AddObjectRef(module, "Resampler", type) < 0) {
        Py_XDECREF(type);
        Py_DECREF(module);
        return NULL;
    }
    Py_DECREF(type);
    return module;
}
