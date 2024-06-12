#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "DFT.h"
#include "DITA.h"
#include "tool_funcs.h"


int test_dbsizes(std::string& filename_dataset, const std::string& measure,
                int is_gpu, unsigned short fanout, unsigned short num_pivots, 
                int n_db, int n_q, int min_trajlen, int max_trajlen,
                int topk, int c) {
    Timer _timer;
    double buildtime, querytime, buildmem;
    std::string dataset_name = filename_dataset.substr(filename_dataset.find_last_of('/')+1);
    dataset_name = dataset_name.substr(0, dataset_name.find('_'));
    long _ram_init = GetProcRAM("knnquery_exp");
    std::cout << "[Flag] start. RAM=" << _ram_init << std::endl;

    if (!measure.compare("DTW") || !measure.compare("ERP")) {
        DITA* dita = new DITA(fanout, num_pivots);
        buildtime = dita->Build(filename_dataset, n_db, min_trajlen, max_trajlen);
        buildmem = GetProcRAM("knnquery_exp") - _ram_init;
        std::cout << "[Flag]DITA build done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") << std::endl;

        std::vector< std::vector<std::pair<float, unsigned int> > > knn_results;
        querytime = dita->Knn(filename_dataset, n_q, measure, is_gpu, min_trajlen, max_trajlen, topk, c, knn_results);

        std::cout << "[Flag]DITA knn done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") 
                << ", retrived trajs=" << g_counter.Get(CNT_KNN_TRAJ_RETRIVED) << std::endl;

        delete dita;
        dita = nullptr;
    }
    else if (!measure.compare("Hausdorff") || !measure.compare("Frechet")) {
        DFT* dft = new DFT(fanout);
        buildtime = dft->Build(filename_dataset, n_db, min_trajlen, max_trajlen);
        buildmem = GetProcRAM("knnquery_exp") - _ram_init;
        std::cout << "[Flag]DFT build done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") << std::endl;
        dft->Verify();
        
        std::vector< std::vector<std::pair<float, unsigned int> > > knn_results;
        querytime = dft->Knn(filename_dataset, n_q, measure, is_gpu, min_trajlen, max_trajlen, topk, c, knn_results);
        // dft->Knn_bruteforce(filename_dataset, n_q, topk);
        // dft->Knn_parallel(filename_dataset, n_q, topk, c);
        
        std::cout << "[Flag]DFT knn done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") 
                    << ", retrived trajs=" << g_counter.Get(CNT_KNN_TRAJ_RETRIVED) << std::endl;
        delete dft;
        dft = nullptr;
    }

    std::cout << "[EXPFlag]exp=knn_dbset_sizes,fn=" << measure << ",dataset=" << dataset_name 
                << ",gpu=" << is_gpu << ",dbset_size=" << n_db << ",qset_size=" << n_q 
                << ",min_traj_len=" << min_trajlen << ",max_traj_len=" << max_trajlen
                << ",buildtime=" << buildtime << ",buildmem=" << buildmem
                << ",querytime=" << querytime << ",querymem=0.0,querytimeemb=0.0" << std::endl;
    return 0;
}


int test_qnumpoints(std::string& filename_dataset, const std::string& measure,
                int is_gpu, unsigned short fanout, unsigned short num_pivots, 
                int n_db, int n_q, int min_trajlen, int max_trajlen,
                int topk, int c) {
    
    Timer _timer;
    double buildtime, querytime, buildmem;
    std::string dataset_name = filename_dataset.substr(filename_dataset.find_last_of('/')+1);
    dataset_name = dataset_name.substr(0, dataset_name.find('_'));
    long _ram_init = GetProcRAM("knnquery_exp");
    std::cout << "[Flag] start. RAM=" << _ram_init << std::endl;

    if (!measure.compare("DTW") || !measure.compare("ERP")) {
        DITA* dita = new DITA(fanout, num_pivots);
        buildtime = dita->Build(filename_dataset, n_db, min_trajlen, max_trajlen);
        buildmem = GetProcRAM("knnquery_exp") - _ram_init;
        std::cout << "[Flag]DITA build done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") << std::endl;

        int qnumpoints[] = {20, 200, 400, 800, 1600};
        for (size_t i = 0; i <= 3; i++) {
            int qmintrajlen = qnumpoints[i];
            int qmaxtrajlen = qnumpoints[i+1];
            std::vector< std::vector<std::pair<float, unsigned int> > > knn_results;
            querytime = dita->Knn(filename_dataset, n_q, measure, is_gpu, qmintrajlen, qmaxtrajlen, topk, c, knn_results);
            std::cout << "[Flag]DITA knn done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") 
                        << ", retrived trajs=" << g_counter.Get(CNT_KNN_TRAJ_RETRIVED) << std::endl;

            std::cout << "[EXPFlag]exp=knn_qset_numpoints,fn=" << measure << ",dataset=" << dataset_name 
                        << ",gpu=" << is_gpu << ",dbset_size=" << n_db << ",qset_size=" << n_q 
                        << ",min_traj_len=" << qmintrajlen << ",max_traj_len=" << qmaxtrajlen
                        << ",buildtime=" << buildtime << ",buildmem=" << buildmem
                        << ",querytime=" << querytime << ",querymem=0.0,querytimeemb=0.0" << std::endl;
        }
        delete dita;
        dita = nullptr;
    }
    else if (!measure.compare("Hausdorff") || !measure.compare("Frechet")) {
        DFT* dft = new DFT(fanout);
        buildtime = dft->Build(filename_dataset, n_db, min_trajlen, max_trajlen);
        buildmem = GetProcRAM("knnquery_exp") - _ram_init;
        std::cout << "[Flag]DFT build done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") << std::endl;
        dft->Verify();
        
        int qnumpoints[] = {20, 200, 400, 800, 1600};
        for (size_t i = 0; i <= 3; i++) {
            int qmintrajlen = qnumpoints[i];
            int qmaxtrajlen = qnumpoints[i+1];

            std::vector< std::vector<std::pair<float, unsigned int> > > knn_results;
            querytime = dft->Knn(filename_dataset, n_q, measure, is_gpu, qmintrajlen, qmaxtrajlen, topk, c, knn_results);
            // dft->Knn_bruteforce(filename_dataset, n_q, topk);
            // dft->Knn_parallel(filename_dataset, n_q, topk, c);
            std::cout << "[Flag]DFT knn done. @=" << _timer << ", RAM=" << GetProcRAM("knnquery_exp") 
                        << ", retrived trajs=" << g_counter.Get(CNT_KNN_TRAJ_RETRIVED) << std::endl;
            std::cout << "[EXPFlag]exp=knn_qset_numpoints,fn=" << measure << ",dataset=" << dataset_name 
                        << ",gpu=" << is_gpu << ",dbset_size=" << n_db << ",qset_size=" << n_q 
                        << ",min_traj_len=" << qmintrajlen << ",max_traj_len=" << qmaxtrajlen
                        << ",buildtime=" << buildtime << ",buildmem=" << buildmem
                        << ",querytime=" << querytime << ",querymem=0.0,querytimeemb=0.0" << std::endl;
        }
        delete dft;
        dft = nullptr;
    }

    return 0;
}



//  (make clean && make all) &> result
//  ./knnquery_exp dbset_sizes DTW xian_7_20inf.h5 0 1000000 10 20 200 20 200 50 10  &> result
int main(int argc, char const *argv[]) {

    if (argc < 12) {
        std::cerr << "wrong argc." << std::endl;
        return -1;
    }

    std::string     exp = argv[1]; // dbset_sizes, qset_numpoints
    std::string     measure = argv[2]; // ERP DTW Hausdorff Frechet
    std::string     filename_dataset = argv[3]; 
    int             is_gpu = atoi(argv[4]);
    unsigned short  rtree_fanout = 25; // 100
    int             n_db = atoi(argv[5]); // 10000 100000 1000000
    int             n_q = atoi(argv[6]); // 1000
    int             db_min_trajlen = atoi(argv[7]); // 20
    int             db_max_trajlen = atoi(argv[8]); // 200
    int             q_min_trajlen = atoi(argv[9]); // 20
    int             q_max_trajlen = atoi(argv[10]); // 200
    int             topk = atoi(argv[11]); // 50
    int             c = atoi(argv[12]); // DONT modify 5

    std::string filepath_dataset = "../../data/" + filename_dataset;
    
    Py_Initialize();

    if (!exp.compare("dbset_sizes")) {
        test_dbsizes(filepath_dataset, measure, is_gpu, rtree_fanout, 3, n_db, n_q,
                        db_min_trajlen, db_max_trajlen, topk, c);
    }
    else if (!exp.compare("qset_numpoints")) {
        test_qnumpoints(filepath_dataset, measure, is_gpu, rtree_fanout, 3, n_db, n_q,
                        db_min_trajlen, db_max_trajlen, topk, c);
    }

    Py_Finalize();
    return 0;
}
