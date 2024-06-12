#include <cmath>
#include <limits>
#include <algorithm>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "traj.h"
#include "tool_funcs.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

Point Traj::erp_gpoint(0.0, 0.0);

Traj::Traj() {

}

Traj::~Traj() {
    for (auto& p: this->points) {
        delete p;
        p = nullptr;
    }

    for (auto& e: this->edges) {
        delete e;
        e = nullptr;
    }
}


size_t Traj::NumPoints() const{
    return this->points.size();
}


size_t Traj::NumEdges() const{
    return this->edges.size();
}


float Traj::Length() const{
    float l = 0.0;
    for (auto e: this->edges) {
        l += e->Length();
    }
    return l;
}

float Traj::HausdorffDistance(Traj* b) {
    // by ychang

    size_t len_a = this->NumPoints();
    size_t len_b = b->NumPoints();

    float** mdist = new float* [len_a];
    float a_dist[len_a - 1];
    float b_dist[len_b - 1];

    for (size_t i = 0; i < this->NumPoints(); ++i) {
        mdist[i] = new float[len_b];
        for (size_t j = 0; j < b->NumPoints(); ++j) {
            mdist[i][j] = L2(this->points[i]->x, this->points[i]->y, b->points[j]->x, b->points[j]->y);
        }
    }
    
    for (size_t i = 0; i < this->NumEdges(); ++i) {
        a_dist[i] = this->edges[i]->Length();
    }
    for (size_t j = 0; j < b->NumEdges(); ++j) {
        b_dist[j] = b->edges[j]->Length();
    }

    float rtn = std::max(Traj::HausdorffDistance_directed(this, b, a_dist, b_dist, mdist, false),
                    Traj::HausdorffDistance_directed(b, this, b_dist, a_dist, mdist, true));

    for (int i = 0; i < len_a; i++) {
        delete[] mdist[i];
    }
    delete[] mdist;
    return rtn;
}



float Traj::HausdorffDistance_directed(Traj* a, Traj* b, float* a_dist, float* b_dist,
                                        float** mdist, bool rotate_mdist) {
    
    float dh = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < a->NumPoints(); ++i) {
        // dist of point to traj
        float dpt = std::numeric_limits<float>::max();
        for (size_t j = 0; j < b->NumEdges(); ++j) {
            float dpl = 0.0;
            float px = a->points[i]->x;
            float py = a->points[i]->y;
            float p1x = b->edges[j]->start->x;
            float p1y = b->edges[j]->start->y;
            float p2x = b->edges[j]->end->x;
            float p2y = b->edges[j]->end->y;
            float dps1 = !rotate_mdist ? mdist[i][j] : mdist[j][i];
            float dps2 = !rotate_mdist ? mdist[i][j+1] : mdist[j+1][i];
            float ds = b_dist[j];
            
            if (p1x == p2x && p1y == p2y) {
                dpl = dps1;
            }
            else {
                float x_diff = p2x - p1x;
                float y_diff = p2y - p1y;
                float u1 = (((px - p1x) * x_diff) + ((py - p1y) * y_diff));
                float u = u1 / (ds * ds);

                if (u < 0.00001 || u > 1) {
                    dpl = std::min(dps1, dps2);
                }
                else {
                    float ix = p1x + u * x_diff;
                    float iy = p1y + u * y_diff;
                    dpl = L2(px, py, ix, iy);
                }
            }
            dpt = std::min(dpt, dpl);
        }
        dh = std::max(dh, dpt);
    }
   return dh;
}


float Traj::Frechet(Traj* b) {
    int n1 = this->NumPoints();
    int n2 = b->NumPoints();

    float fmax = std::numeric_limits<float>::max();
    size_t ncol = (n2+1);
    int flag = 1;

    float* dp = new float[2*ncol];
    std::fill_n(dp, 2*ncol, fmax);
    dp[0] = 0.0;
    dp[ncol] = fmax;

    for (size_t i = 1; i < n1+1; ++i) {
        for (size_t j = 1; j < n2+1; ++j) {
            float d = L2(this->points[i-1]->x, this->points[i-1]->y, b->points[j-1]->x, b->points[j-1]->y);
            float s = std::min(std::min(dp[j-1 + flag * ncol], dp[j-1 + (1-flag) * ncol]), dp[j + (1-flag) * ncol]);
            dp[j + flag * ncol] = std::max(d, s);
        }   
        flag = 1 - flag;
        dp[0] = fmax;
    }
    float rtn = dp[n2 + (1-flag) * ncol];
    delete dp;
    return rtn;
}


float Traj::DTW(Traj* b) {
    int n1 = this->NumPoints();
    int n2 = b->NumPoints();

    float fmax = std::numeric_limits<float>::max();
    size_t ncol = (n2+1);
    int flag = 1;

    float* dp = new float[2*ncol];
    std::fill_n(dp, 2*ncol, fmax);
    dp[0] = 0.0;
    dp[ncol] = fmax;

    for (size_t i = 1; i < n1+1; ++i) {
        for (size_t j = 1; j < n2+1; ++j) {
            float d = L2(this->points[i-1]->x, this->points[i-1]->y, b->points[j-1]->x, b->points[j-1]->y);
            dp[j + flag * ncol] = d + std::min(std::min(dp[j-1 + flag * ncol], dp[j-1 + (1-flag) * ncol]), dp[j + (1-flag) * ncol]);
        }
        flag = 1 - flag;
        dp[0] = fmax;
    }
    
    float rtn = dp[n2 + (1-flag) * ncol];
    delete dp;
    return rtn;
}


float Traj::ERP(Traj* b, Point& p) {
    int n1 = this->NumPoints();
    int n2 = b->NumPoints();

    float* gt0_dist = new float[n1];
    float* gt1_dist = new float[n2];
    float gt0_sum = 0;
    float gt1_sum = 0;

    for (size_t i = 0; i < this->points.size(); ++i) {
        gt0_dist[i] = L2(this->points[i]->x, this->points[i]->y, p.x, p.y);
        gt0_sum += gt0_dist[i];
    }

    for (size_t i = 0; i < b->points.size(); ++i) {
        gt1_dist[i] = L2(b->points[i]->x, b->points[i]->y, p.x, p.y);
        gt1_sum += gt1_dist[i];
    }

    float* mdist = new float[n1*n2];
    for (size_t i = 0; i < this->points.size(); ++i) {
        Point* p1 = this->points[i];
        for (size_t j = 0; j < b->points.size(); ++j) {
            Point* p2 = b->points[j];
            mdist[i*n2+j] = sqrt( pow(p1->x - p2->x, 2) + pow(p1->y - p2->y, 2) );
        }
    }

    size_t ncol = (n2+1);
    int flag = 1;

    float* dp = new float[2*ncol];
    std::fill_n(dp, 2*ncol, gt1_sum);
    dp[0] = 0.0;
    dp[ncol] = gt0_sum;

    for (size_t i = 1; i < n1+1; ++i) {
        for (size_t j = 1; j < n2+1; ++j) {
            float derp0 = dp[j + (1-flag) * ncol] + gt0_dist[i-1];
            float derp1 = dp[j-1 + flag * ncol] + gt1_dist[j-1];
            float derp01 = dp[j-1 + (1-flag) * ncol] + mdist[(i-1)*n2+j-1];
            dp[j + flag * ncol] = std::min(std::min(derp0, derp1), derp01);
        }
        flag = 1 - flag;
        dp[0] = gt0_sum;
    }

    float rtn = dp[n2 + (1-flag) * ncol];
    delete gt0_dist;
    delete gt1_dist;
    delete mdist;
    delete dp;
    
    return rtn;
}


int Traj::TrajSimiGPU(std::vector<Traj*> trajs1, std::vector<Traj*> trajs2,
                        const std::string& measure, std::vector<float>& dists) {

    // Py_Initialize(); 
    // Py_Initialize and Py_Finalize can be called only once.
    // Thus, they are moved into main function.
    // THis is a common issue of python c-api and numpy.
    // See https://github.com/numpy/numpy/issues/8097
    import_array();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('..')");
    PyRun_SimpleString("sys.path.append('../..')");
    PyRun_SimpleString("import core");
    
    PyObject* pymodule = nullptr;
    PyObject* pyfunc = nullptr;
    if (!measure.compare("DTW")) {
        pymodule = PyImport_ImportModule("core.dtw_parallel_long2");
        pyfunc = PyObject_GetAttrString(pymodule, "dtw");
    }
    else if (!measure.compare("ERP")) {
        pymodule = PyImport_ImportModule("core.erp");
        pyfunc = PyObject_GetAttrString(pymodule, "erp");
    }
    else if (!measure.compare("Hausdorff")) {
        pymodule = PyImport_ImportModule("core.hausdorff_long2");
        pyfunc = PyObject_GetAttrString(pymodule, "hausdorff");
    }
    else if (!measure.compare("Frechet")) {
        pymodule = PyImport_ImportModule("core.discret_frechet");
        pyfunc = PyObject_GetAttrString(pymodule, "dfrechet");
    }
    // std::cout << "pymodule, pyfunc=" <<pymodule << ',' << pyfunc << std::endl;

    int max_trajlen = 0;
    int n = trajs1.size();

    npy_intp dims_len[1] = {n};
    PyObject* arr_len = PyArray_Zeros(1, dims_len, PyArray_DescrFromType(NPY_INT), 0);
    int* dptr_len = (int *)PyArray_DATA(arr_len);
    PyObject* arr2_len = PyArray_Zeros(1, dims_len, PyArray_DescrFromType(NPY_INT), 0);
    int* dptr2_len = (int *)PyArray_DATA(arr2_len);

    for(int i = 0; i < n; i++) {
        auto t1 = trajs1[i];
        auto t2 = trajs2[i];
        dptr_len[i] = t1->NumPoints();
        dptr2_len[i] = t2->NumPoints();
        if (dptr_len[i] > max_trajlen) 
            { max_trajlen = dptr_len[i]; }
        if (dptr2_len[i] > max_trajlen) 
            { max_trajlen = dptr2_len[i]; }
    }
    
    npy_intp dims[3] = {n, max_trajlen, 2};
    PyObject* arr = PyArray_Zeros(3, dims, PyArray_DescrFromType(NPY_DOUBLE), 0);
    double* dptr = (double *)PyArray_DATA(arr);
    PyObject* arr2 = PyArray_Zeros(3, dims, PyArray_DescrFromType(NPY_DOUBLE), 0);
    double* dptr2 = (double *)PyArray_DATA(arr2);

    for(int i = 0; i < n; i++) {
        auto t1 = trajs1[i];
        auto t2 = trajs2[i];

        for(int j = 0; j < t1->NumPoints(); j++){
            dptr[ i*dims[2]*dims[1] + j * dims[2] + 0] = t1->points[j]->x;
            dptr[ i*dims[2]*dims[1] + j * dims[2] + 1] = t1->points[j]->y;
        }
        for(int j = 0; j < t2->NumPoints(); j++){
            dptr2[ i*dims[2]*dims[1] + j * dims[2] + 0] = t2->points[j]->x;
            dptr2[ i*dims[2]*dims[1] + j * dims[2] + 1] = t2->points[j]->y;
        }
    }

    PyObject* pyargs = nullptr;
    PyObject* pyrtn = nullptr;
    if (!measure.compare("ERP")) {
        npy_intp dims_gpoint[2] = {1, 2};
        PyObject* arr_gpoint = PyArray_Zeros(1, dims_gpoint, PyArray_DescrFromType(NPY_DOUBLE), 0);
        double* dptr_gpoint = (double *)PyArray_DATA(arr_gpoint);
        dptr_gpoint[0] = 0.0;
        dptr_gpoint[1] = 0.0;// Traj::erp_gpoint.y;
        
        pyargs = PyTuple_New(5);
        PyTuple_SetItem(pyargs, 0, arr);
        PyTuple_SetItem(pyargs, 1, arr_len);
        PyTuple_SetItem(pyargs, 2, arr2);
        PyTuple_SetItem(pyargs, 3, arr2_len);
        PyTuple_SetItem(pyargs, 4, arr_gpoint);
        pyrtn = PyEval_CallObject(pyfunc, pyargs);
    }
    else {
        pyargs = PyTuple_New(4);
        PyTuple_SetItem(pyargs, 0, arr);
        PyTuple_SetItem(pyargs, 1, arr_len);
        PyTuple_SetItem(pyargs, 2, arr2);
        PyTuple_SetItem(pyargs, 3, arr2_len);
        pyrtn = PyEval_CallObject(pyfunc, pyargs);
    }

    // std::cout << (pyrtn == NULL) << " " << PyArray_Check(pyrtn) << std::endl;
    PyArrayObject *pyrtn_array;
    if (!PyArg_Parse(pyrtn, "O!", &PyArray_Type, &pyrtn_array)) {
        std::cout << "PyArg return value parsed failed, " << pyrtn_array << std::endl;
    }
    const double* pyrtn_p = (const double*) pyrtn_array->data;
    int pyrtn_len = pyrtn_array->dimensions[0];
    // std::cout << pyrtn_len << std::endl;
    // std::cout << *(pyrtn_p+1) << std::endl;
    for (int i = 0; i < pyrtn_len; ++i) {
        dists.push_back( *(pyrtn_p+1) );
    }

 
    Py_DECREF(pyargs);
    Py_DECREF(pyrtn_array);
    // Py_Finalize();
    return 0;
}


std::istream& operator>>(std::istream& in, Traj& traj) {

    in.precision(7);
    size_t points_len = 0;
    double x = 0.0, y =0.0;
    long t = 0;

    in >> traj.id >> points_len;
    for (size_t i = 0; i < points_len; ++i) {
        in >> x >> y;
        Point* p = new Point(x, y);
        
        if (i > 0) {
            Edge* e = new Edge(traj.points.back(), p, traj.id);
            traj.edges.push_back(e);
        }
        traj.points.push_back(p);
    }
    return in;
}

std::ostream& operator<<(std::ostream& out, Traj& traj) {
    const Points& points = traj.points;
    
    out.precision(7);
    out << traj.id << " " << traj.NumPoints();
    for (size_t i = 0; i < traj.NumPoints(); ++i) {
        out << " " << points[i]->x << " " << points[i]->y;
    }
    return out;
}
