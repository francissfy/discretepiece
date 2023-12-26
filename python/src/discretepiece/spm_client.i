%module discretepiece
%{
#include "spm_client.h"
%}
%include "std_vector.i"

namespace std {
    %template(IntVector) vector<int>;
}

%include "spm_client.h"