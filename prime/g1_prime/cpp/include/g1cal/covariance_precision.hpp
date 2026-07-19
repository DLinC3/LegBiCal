// Strict diagonal covariance/precision contract for the first grouped model.
#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace g1cal
{

struct DiagonalCovariancePrecision
{
    bool enabled = false;
    Eigen::VectorXd p0; // 70-dim motion tangent precision
    Eigen::VectorXd q;  // 35-dim generalized-force precision
    Eigen::VectorXd r;  // 70-dim measurement tangent precision
    std::string config_hash;

    void validate(const int ndx, const int nu) const
    {
        if (!enabled)
            return;
        if (p0.size() != ndx || r.size() != ndx || q.size() != nu)
            throw std::runtime_error("covariance precision dimensions invalid");
        if (!p0.allFinite() || !q.allFinite() || !r.allFinite() ||
            (p0.array() <= 0.).any() || (q.array() <= 0.).any() ||
            (r.array() <= 0.).any())
            throw std::runtime_error("covariance precision must be finite positive");
        if (config_hash.empty())
            throw std::runtime_error("covariance config hash is required");
    }
};

inline Eigen::VectorXd parse_precision_row(const std::string &line,
                                           const std::string &expected_label)
{
    std::stringstream ss(line);
    std::string field;
    if (!std::getline(ss, field, ',') || field != expected_label)
        throw std::runtime_error("expected precision row " + expected_label);
    std::vector<double> values;
    while (std::getline(ss, field, ','))
    {
        if (!field.empty())
            values.push_back(std::stod(field));
    }
    Eigen::VectorXd result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        result[static_cast<Eigen::Index>(i)] = values[i];
    return result;
}

inline DiagonalCovariancePrecision
load_covariance_precision(const std::string &path, const int ndx, const int nu)
{
    std::ifstream input(path);
    if (!input.is_open())
        throw std::runtime_error("cannot open covariance precision file: " + path);
    DiagonalCovariancePrecision out;
    out.enabled = true;
    std::string line;
    std::vector<std::string> rows;
    while (std::getline(input, line))
    {
        if (line.rfind("# config_hash=", 0) == 0)
            out.config_hash = line.substr(std::string("# config_hash=").size());
        else if (!line.empty() && line[0] != '#')
            rows.push_back(line);
    }
    if (rows.size() != 3)
        throw std::runtime_error("precision file must contain p0,q,r rows");
    out.p0 = parse_precision_row(rows[0], "p0");
    out.q = parse_precision_row(rows[1], "q");
    out.r = parse_precision_row(rows[2], "r");
    out.validate(ndx, nu);
    return out;
}

} // namespace g1cal
