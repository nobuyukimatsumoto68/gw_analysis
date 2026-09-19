#pragma once


#include <functional>

using Idx = std::size_t;

template<typename T>
class JackknifeSimp {
public:
  std::vector<T> jack_avg;
  T mean;
  T var;

  Idx nbins;
  Idx len;

  JackknifeSimp( const Idx nbins, const Idx len )
    : jack_avg( nbins, T::Zero(len) )
    , nbins(nbins)
    , len(len)
  {
  }

  void finalize(){
    this->mean = T::Zero(len);
    this->var = T::Zero(len);

    for(int i=0; i<nbins; i++) mean += jack_avg[i];
    mean /= nbins;
    for(int i=0; i<nbins; i++) var += (jack_avg[i] - mean).array().square().matrix();
    var *= 1.0*(nbins-1)/nbins;
  }

  void eps_trick( const double eps ){
    for(int i=0; i<nbins; i++) {
      jack_avg[i] = mean + eps*(jack_avg[i]-mean);
    }
  }

  void eps_trick( const double eps, const T mn ){
    for(int i=0; i<nbins; i++) {
      jack_avg[i] = mn + eps*(jack_avg[i]-mn);
    }
  }

  // void write( const T1& dat, const std::string& filepath, const Idx size ) const {
  //   std::ofstream of( filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  //   if(!of) assert(false);

  //   double tmp = 0;
  //   for(Idx i=0; i<size; i++){
  //     tmp = dat[i];
  //     of.write( (char*) &tmp, sizeof(double) );
  //   }
  //   of.close();
  // }

  // void read( T1& dat, const std::string& filepath, const Idx size ) {
  //   std::ifstream ifs( filepath, std::ios::in | std::ios::binary );
  //   if(!ifs) assert(false);

  //   double tmp;
  //   for(Idx i=0; i<size; i++){
  //     ifs.read((char*) &tmp, sizeof(double) );
  //     dat[i] = tmp;
  //   }
  //   ifs.close();
  // }


};
