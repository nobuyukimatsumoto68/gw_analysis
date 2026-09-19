#pragma once


#include <functional>

template<typename T1>
class JackknifeSimp {
public:
  std::vector<T1> jack_avg;
  T1 mean;
  T1 var;

  int nbins;

  JackknifeSimp(const int nbins)
    : nbins(nbins)
    , jack_avg(nbins, T1::Zero())
  {}

  void finalize(){
    assert(jack_avg.size()==nbins);

    this->mean = T1::Zero();
    this->var = T1::Zero();

    for(int i=0; i<nbins; i++) mean += jack_avg[i];
    mean /= nbins;
    for(int i=0; i<nbins; i++) var += (jack_avg[i] - mean).array().square().matrix();
    var *= 1.0*(nbins-1)/nbins;
  }

};
