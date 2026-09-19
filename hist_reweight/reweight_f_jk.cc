// #include "reweight.h"
#include "reweight2.h"

#include <regex>

// #define NOSIGN

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;
  // omp_set_num_threads(threads);

  // -------------------------------------
  // for reading data
  const int conf_min0=atoi(argv[1]);
  const std::string base_dir(argv[2]);
  const std::string basedir(argv[3]);
  const std::string basedir2c(argv[4]);
  const std::string basedir2h(argv[5]);
  const std::string mass(argv[6]);
  const int conf_max0=atoi(argv[7]);
  const int nbin_global=atoi(argv[8]);
  int nbeta=atoi(argv[9]);
  int runtype=atoi(argv[10]);

  std::cout << "nbeta = " << nbeta << std::endl;
  std::cout << "basedir = " << basedir << std::endl;

  int type=1;
  if(basedir2c==basedir2h) {
    if(mass=="0p1000" || mass=="0p4000") type=2;
    else if(mass=="0p0500" || mass=="0p0100" || mass=="0p2000" || mass=="0p3000") type=4;
    else if(mass=="0") type=6;
  }
  else if(mass=="0") type = 0;

  // -----------------------

  std::vector<std::string> betas;
  std::vector<double> betas_double;
  {
    for(int i=11; i<11+nbeta; i++) {
      std::string str(argv[i]);
      betas.push_back(str);

      std::regex to_replace("p");
      str = std::regex_replace(str, to_replace, ".");
      std::cout << "str replaced = " << str << std::endl;

      // std::replace( str, "p", "." );
      betas_double.push_back(std::stod(str));
    }
  }

  using Impl = PeriodicGimplR;

// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
  for(std::string beta : betas){
    std::cout << GridLogMessage << "beta = " << beta << std::endl;
    EnsembleOfObs ensR;
    ensR.beta = beta;
    ensR.base_dir=base_dir;
    if(type>1) ensR.id=get_configname(beta, mass, type+1);
    else ensR.id=get_configname(beta, mass, type);

    EnsembleOfObs ensI;
    ensI.beta = beta;
    ensI.base_dir=base_dir;
    if(type>1) ensI.id=get_configname(beta, mass, type+1);
    else ensI.id=get_configname(beta, mass, type);

    std::string path_ens = basedir+ensR.id+".bin";
    // std::cout << "path_ens = " << path_ens << std::endl;
    if( std::filesystem::exists( path_ens ) && (runtype==0) ) {
      continue;
    }

    // ---------------------------
    // cold start
    // ---------------------------

    ensR.id=get_configname(beta, mass, type);
    ensI.id=get_configname(beta, mass, type);

    for(int conf=conf_min0; conf<conf_max0; conf++){
      std::string pathO;
      if(type>1) pathO = basedir2c+ensR.id+std::to_string(conf)+".h5";
      else pathO = basedir2c+ensR.id+std::to_string(conf)+".bin";
      // std::cout << "pathO = " << pathO << std::endl;
      if( !std::filesystem::exists( pathO ) ) continue; // assert(false);

      Obs obsR, obsI;
#pragma omp critical
      {
        std::unique_ptr<Hdf5Reader> WR;
        WR = std::make_unique<Hdf5Reader>( pathO );
        read(*WR, "obs", obsR );
        obsI = obsR;
        std::vector<ComplexD> all_polyakov;
        read(*WR, "polyakov", all_polyakov );
        obsR.O = real( all_polyakov.back() );
        obsI.O = imag( all_polyakov.back() );
      }
      ensR.Os.push_back(obsR.O);
      ensR.Osqs.push_back(obsR.O*obsR.O);
      ensR.energies.push_back(obsR.energy);
      //
      ensI.Os.push_back(obsI.O);
      ensI.Osqs.push_back(obsI.O*obsI.O);
      ensI.energies.push_back(obsI.energy);
    } // end for conf

    // ---------------------------
    // hot start
    // ---------------------------

    if(type>1) {
      ensR.id=get_configname(beta, mass, type+1);
      ensI.id=get_configname(beta, mass, type+1);
    }

    for(int conf=conf_min0; conf<conf_max0; conf++){
      // std::string pathO = basedir2h+ensR.id+std::to_string(conf)+".bin";
      std::string pathO;
      if(type>1) pathO = basedir2c+ensR.id+std::to_string(conf)+".h5";
      else pathO = basedir2c+ensR.id+std::to_string(conf)+".bin";
      if( !std::filesystem::exists( pathO ) ) continue; // assert(false);

      Obs obsR, obsI;
#pragma omp critical
      {
        std::unique_ptr<Hdf5Reader> WR;
        WR = std::make_unique<Hdf5Reader>( pathO );
        read(*WR, "obs", obsR );
        obsI = obsR;
        std::vector<ComplexD> all_polyakov;
        read(*WR, "polyakov", all_polyakov );
        obsR.O = real( all_polyakov.back() );
        obsI.O = imag( all_polyakov.back() );
      }

      ensR.Os.push_back(obsR.O);
      ensR.Osqs.push_back(obsR.O*obsR.O);
      ensR.energies.push_back(obsR.energy);
      //
      ensI.Os.push_back(obsI.O);
      ensI.Osqs.push_back(obsI.O*obsI.O);
      ensI.energies.push_back(obsI.energy);
    } // end for conf

    ensR.nbin = nbin_global;
    ensR.binsize = ensR.size()/nbin_global;
    ensR.nconf = ensR.nbin*ensR.binsize;
    ensI.nbin = nbin_global;
    ensI.binsize = ensR.size()/nbin_global;
    ensI.nconf = ensI.nbin*ensI.binsize;

    { // trimming
      const int start = ensR.nconf;
      ensR.energies.erase( ensR.energies.begin()+start, ensR.energies.end() );
      ensR.Os.erase( ensR.Os.begin()+start, ensR.Os.end() );
      ensR.Osqs.erase( ensR.Osqs.begin()+start, ensR.Osqs.end() );
    }
    { // trimming
      const int start = ensI.nconf;
      ensI.energies.erase( ensI.energies.begin()+start, ensI.energies.end() );
      ensI.Os.erase( ensI.Os.begin()+start, ensI.Os.end() );
      ensI.Osqs.erase( ensI.Osqs.begin()+start, ensI.Osqs.end() );
    }

    std::cout << GridLogMessage << "beta = " << beta << " size = " << ensR.size() << " nbin = " << ensR.nbin << " nconf = " << ensR.nconf << std::endl;

#pragma omp critical
    {
      {
        std::unique_ptr<Hdf5Writer> WR;
        WR = std::make_unique<Hdf5Writer>( path_ens );
        write(*WR, "ensR", ensR );
        write(*WR, "ensI", ensI );
      }
    }
  } // for end beta



  // ################################################################ //
  std::cout << GridLogMessage << "run2 part v.2" << std::endl;

  if(type>1) type+=1;

  {
    for(int jdrop=0; jdrop<nbeta; jdrop++){
      std::cout << "jdrop = " << jdrop << std::endl;
      const int nbin=nbin_global; // ensembles0[jdrop].nbin;
      for(int ibin=0; ibin<nbin; ibin++){
        std::cout << "ibin = " << ibin << std::endl;

        std::vector<EnsembleOfObs> ensembles0( nbeta );

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
        for(int j=0; j<nbeta; j++){
          const std::string beta = betas[j];
          std::string id_str = get_configname(beta, mass, type);
          std::string path = basedir+id_str+".bin";

#pragma omp critical
          {
            std::unique_ptr<Hdf5Reader> WR;
            WR = std::make_unique<Hdf5Reader>( path );
            read(*WR, "ensR", ensembles0[j] );
          }
        }

        // std::vector<EnsembleOfObs> ensembles(nbeta);
        // for(int i=0; i<nbeta; i++) ensembles[i] = ensembles0[i];
        // std::cout << GridLogMessage << "debug. pt1" << std::endl;
        ensembles0[jdrop].jackknife_drop( nbin, ibin );

        // {
        //   char id[200];
        //   std::sprintf(id, "m%s", mass.data());
        //   std::string id_str = id;
        //   std::string path = basedir+id_str+"_binsize.txt";

        //   std::ofstream file(path, std::ios::trunc);
        //   for(int j=0; j<nbeta; j++){
        //     file << ensembles0[j].binsize << std::endl;
        //   }
        // }

        char id[200];
        std::sprintf(id, "m%s", mass.data());
        std::string id_str = id;
        id_str = id_str+"_jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
        // id_str = id_str+"_nojk";

        // -------------------
        FreeEnergies fs(nbeta);

        {
          std::string id_str2 = id;
          id_str2 = id_str2+"_nojk";
          std::string filename2=basedir+id_str2+"fs.bin";
          std::cout << "filename2 = " << filename2 << std::endl;
          if( std::filesystem::exists( filename2 ) ){
#pragma omp critical
            {
              std::unique_ptr<Hdf5Reader> WR;
              WR = std::make_unique<Hdf5Reader>( filename2 );
              read(*WR, "f", fs.f );

              std::cout << "fs = ";
              for(auto elem : fs.f) std::cout << elem << " ";
              std::cout << std::endl;
            }
            fs.fnew = fs.f;
            fs.fold = fs.f;
          }
        }

        std::vector<std::vector<RealD>> v_energies(nbeta);
        for(int k=0; k<nbeta; k++) v_energies[k] = ensembles0[k].energies;
        run_multihistogram( betas_double, v_energies, fs, basedir+id_str+"fs.bin", threads );

        std::cout << "fs = ";
        for(auto elem : fs.f) std::cout << elem << " ";
        std::cout << std::endl;
        std::cout << GridLogMessage << "pathf = " << basedir+id_str+"fs.bin" << std::endl;

#pragma omp critical
        {
          const std::string path = basedir+id_str+"fs.bin";
          if( std::filesystem::exists( path ) ) std::filesystem::remove( path );
          std::unique_ptr<Hdf5Writer> WR;
          WR = std::make_unique<Hdf5Writer>( path );
          write(*WR, "f", fs.f );
        }

      } // end for ibin
    } // end for jdrop
  }

  Grid_finalize();
}
