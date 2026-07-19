#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Compile the already-validated runner into this module and expose its entry
// point without maintaining a second solver implementation.  Renaming main is
// local to this translation unit.
#define main g1_motion_fie_entry
#include "../apps/g1_motion_fie.cpp"
#undef main

namespace py = pybind11;

namespace
{

class StreamCapture
{
public:
    StreamCapture()
        : old_out_(std::cout.rdbuf(output_.rdbuf())),
          old_err_(std::cerr.rdbuf(output_.rdbuf()))
    {
    }

    ~StreamCapture()
    {
        std::cout.rdbuf(old_out_);
        std::cerr.rdbuf(old_err_);
    }

    std::string str() const { return output_.str(); }

private:
    std::ostringstream output_;
    std::streambuf *old_out_;
    std::streambuf *old_err_;
};

py::tuple run_motion_fie(const std::vector<std::string> &arguments)
{
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.emplace_back("g1_motion_fie");
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (std::string &argument : storage)
        argv.push_back(argument.data());

    StreamCapture capture;
    const int return_code =
        g1_motion_fie_entry(static_cast<int>(argv.size()), argv.data());
    return py::make_tuple(return_code, capture.str());
}

} // namespace

PYBIND11_MODULE(_g1cal_cpp, module)
{
    module.doc() = "In-process fixed-inertia G1 PRIME motion-FIE runner";
    module.def("run_motion_fie", &run_motion_fie, py::arg("arguments"));
}
