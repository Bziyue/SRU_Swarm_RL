#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Eigen>

#include "SplineTrajectory/SplineTrajectory.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/spline_sfc_optimizer.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace
{

struct Vec3f
{
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
};

struct Vec3i
{
  int x{0};
  int y{0};
  int z{0};
};

struct Vec4d
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double w{0.0};
};

struct Region
{
  int id{0};
  std::string name;
  std::vector<Vec3f> corners;
  Vec3f center;
  Vec3f bounds_min;
  Vec3f bounds_max;
};

struct NeighborStep
{
  Vec3i delta;
  float cost{0.0F};
};

struct FrontierState
{
  std::int64_t key{0};
  float cost{0.0F};
};

struct FrontierCompare
{
  bool operator()(const FrontierState & a, const FrontierState & b) const
  {
    return a.cost > b.cost;
  }
};

struct DijkstraResult
{
  std::unordered_map<std::int64_t, float> distance;
  std::unordered_map<std::int64_t, std::int64_t> parent;
  std::unordered_set<std::int64_t> reached_targets;
};

struct PathRecord
{
  int source_id{0};
  std::string source_name;
  int target_id{0};
  std::string target_name;
  bool reachable{false};
  std::vector<Vec3f> guide_points;
  std::vector<Vec3f> points;
};

struct TrajectoryRecord
{
  struct CorridorPolyhedronRecord
  {
    std::vector<Vec4d> halfspaces;
    std::vector<Vec3f> vertices;
  };

  int source_id{0};
  std::string source_name;
  int target_id{0};
  std::string target_name;
  bool path_reachable{false};
  bool optimization_succeeded{false};
  std::string failure_reason;
  double optimization_cost{std::numeric_limits<double>::infinity()};
  double duration{0.0};
  int num_segments{0};
  int num_coefficients{0};
  int num_corridor_polyhedra{0};
  std::vector<Vec3f> guide_points;
  std::vector<Vec3f> resampled_path;
  std::vector<double> breakpoints;
  std::vector<Vec3f> coefficients;
  std::vector<CorridorPolyhedronRecord> safe_corridor;
};

constexpr float kEpsilon = 1e-6F;

std::vector<std::string> split(const std::string & text)
{
  std::istringstream stream(text);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

float distance3(const Vec3f & a, const Vec3f & b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float distance3(const Vec3i & a, const Vec3i & b)
{
  const float dx = static_cast<float>(a.x - b.x);
  const float dy = static_cast<float>(a.y - b.y);
  const float dz = static_cast<float>(a.z - b.z);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Vec3f makeCenter(const std::vector<Vec3f> & points)
{
  Vec3f center{};
  if (points.empty()) {
    return center;
  }
  for (const auto & point : points) {
    center.x += point.x;
    center.y += point.y;
    center.z += point.z;
  }
  const float inv = 1.0F / static_cast<float>(points.size());
  center.x *= inv;
  center.y *= inv;
  center.z *= inv;
  return center;
}

Vec3f makeMinCorner(const std::vector<Vec3f> & points)
{
  Vec3f value{
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()};
  for (const auto & point : points) {
    value.x = std::min(value.x, point.x);
    value.y = std::min(value.y, point.y);
    value.z = std::min(value.z, point.z);
  }
  return value;
}

Vec3f makeMaxCorner(const std::vector<Vec3f> & points)
{
  Vec3f value{
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest()};
  for (const auto & point : points) {
    value.x = std::max(value.x, point.x);
    value.y = std::max(value.y, point.y);
    value.z = std::max(value.z, point.z);
  }
  return value;
}

Vec3f hsvToRgb(float h, float s, float v)
{
  const float hh = std::fmod(std::max(h, 0.0F), 1.0F) * 6.0F;
  const int sector = static_cast<int>(std::floor(hh));
  const float frac = hh - static_cast<float>(sector);
  const float p = v * (1.0F - s);
  const float q = v * (1.0F - s * frac);
  const float t = v * (1.0F - s * (1.0F - frac));

  switch (sector) {
    case 0:
      return {v, t, p};
    case 1:
      return {q, v, p};
    case 2:
      return {p, v, t};
    case 3:
      return {p, q, v};
    case 4:
      return {t, p, v};
    default:
      return {v, p, q};
  }
}

geometry_msgs::msg::Point toPoint(const Vec3f & point)
{
  geometry_msgs::msg::Point ros_point;
  ros_point.x = point.x;
  ros_point.y = point.y;
  ros_point.z = point.z;
  return ros_point;
}

std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

std::string sanitizeName(const std::string & name)
{
  std::string sanitized = name;
  for (char & c : sanitized) {
    const bool safe = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
    if (!safe) {
      c = '_';
    }
  }
  return sanitized;
}

Eigen::Vector3d toEigen(const Vec3f & point)
{
  return Eigen::Vector3d(
    static_cast<double>(point.x),
    static_cast<double>(point.y),
    static_cast<double>(point.z));
}

Vec3f fromEigen(const Eigen::Vector3d & point)
{
  return {
    static_cast<float>(point.x()),
    static_cast<float>(point.y()),
    static_cast<float>(point.z())};
}

std::vector<Eigen::Vector3d> toEigenPoints(const std::vector<Vec3f> & points)
{
  std::vector<Eigen::Vector3d> eigen_points;
  eigen_points.reserve(points.size());
  for (const auto & point : points) {
    eigen_points.push_back(toEigen(point));
  }
  return eigen_points;
}

std::string escapeJsonString(const std::string & value)
{
  std::string escaped;
  escaped.reserve(value.size() + 8U);
  for (const char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

class VoxelMap
{
public:
  VoxelMap(float resolution, const Vec3f & origin, int size_x, int size_y, int size_z)
  : resolution_(resolution), origin_(origin), size_x_(size_x), size_y_(size_y), size_z_(size_z)
  {
  }

  std::int64_t pack(const Vec3i & idx) const
  {
    return static_cast<std::int64_t>(idx.x) +
           static_cast<std::int64_t>(size_x_) *
             (static_cast<std::int64_t>(idx.y) + static_cast<std::int64_t>(size_y_) * idx.z);
  }

  Vec3i unpack(std::int64_t key) const
  {
    Vec3i idx;
    idx.x = static_cast<int>(key % size_x_);
    key /= size_x_;
    idx.y = static_cast<int>(key % size_y_);
    key /= size_y_;
    idx.z = static_cast<int>(key);
    return idx;
  }

  bool inBounds(const Vec3i & idx) const
  {
    return idx.x >= 0 && idx.x < size_x_ &&
           idx.y >= 0 && idx.y < size_y_ &&
           idx.z >= 0 && idx.z < size_z_;
  }

  bool isOccupied(const Vec3i & idx) const
  {
    if (!inBounds(idx)) {
      return true;
    }
    return occupied_.find(pack(idx)) != occupied_.end();
  }

  void addOccupied(const Vec3i & idx)
  {
    if (inBounds(idx)) {
      occupied_.insert(pack(idx));
    }
  }

  Vec3i worldToGrid(const Vec3f & point) const
  {
    return {
      static_cast<int>(std::floor((point.x - origin_.x) / resolution_)),
      static_cast<int>(std::floor((point.y - origin_.y) / resolution_)),
      static_cast<int>(std::floor((point.z - origin_.z) / resolution_))
    };
  }

  Vec3f gridToWorld(const Vec3i & idx) const
  {
    return {
      origin_.x + (static_cast<float>(idx.x) + 0.5F) * resolution_,
      origin_.y + (static_cast<float>(idx.y) + 0.5F) * resolution_,
      origin_.z + (static_cast<float>(idx.z) + 0.5F) * resolution_
    };
  }

  const std::unordered_set<std::int64_t> & occupied() const
  {
    return occupied_;
  }

  void setOccupied(std::unordered_set<std::int64_t> occupied)
  {
    occupied_ = std::move(occupied);
  }

  float resolution() const { return resolution_; }
  const Vec3f & origin() const { return origin_; }
  int sizeX() const { return size_x_; }
  int sizeY() const { return size_y_; }
  int sizeZ() const { return size_z_; }

private:
  float resolution_{0.1F};
  Vec3f origin_{};
  int size_x_{0};
  int size_y_{0};
  int size_z_{0};
  std::unordered_set<std::int64_t> occupied_;
};

std::vector<Vec3f> loadBinaryPlyPoints(const std::string & ply_path)
{
  std::ifstream input(ply_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open PLY file: " + ply_path);
  }

  std::string line;
  std::getline(input, line);
  if (line != "ply") {
    throw std::runtime_error("Invalid PLY header in: " + ply_path);
  }

  std::size_t vertex_count = 0;
  bool little_endian = false;
  bool in_vertex_element = false;
  std::vector<std::pair<std::string, int>> properties;

  while (std::getline(input, line)) {
    if (line == "end_header") {
      break;
    }
    const auto tokens = split(line);
    if (tokens.empty()) {
      continue;
    }

    if (tokens[0] == "format") {
      little_endian = tokens.size() >= 2 && tokens[1] == "binary_little_endian";
    } else if (tokens[0] == "element") {
      in_vertex_element = tokens.size() >= 3 && tokens[1] == "vertex";
      if (in_vertex_element) {
        vertex_count = static_cast<std::size_t>(std::stoul(tokens[2]));
      }
    } else if (tokens[0] == "property" && in_vertex_element) {
      if (tokens.size() != 3) {
        throw std::runtime_error("Only scalar PLY properties are supported.");
      }
      const std::string & type = tokens[1];
      int bytes = 0;
      if (type == "char" || type == "uchar") {
        bytes = 1;
      } else if (type == "short" || type == "ushort") {
        bytes = 2;
      } else if (type == "int" || type == "uint" || type == "float") {
        bytes = 4;
      } else if (type == "double") {
        bytes = 8;
      } else {
        throw std::runtime_error("Unsupported PLY property type: " + type);
      }
      properties.emplace_back(tokens[2], bytes);
    }
  }

  if (!little_endian) {
    throw std::runtime_error("Only binary_little_endian PLY is supported.");
  }
  if (vertex_count == 0 || properties.empty()) {
    throw std::runtime_error("Failed to parse vertex properties from PLY.");
  }

  std::size_t stride = 0;
  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  for (const auto & property : properties) {
    if (property.first == "x") {
      x_offset = static_cast<int>(stride);
    } else if (property.first == "y") {
      y_offset = static_cast<int>(stride);
    } else if (property.first == "z") {
      z_offset = static_cast<int>(stride);
    }
    stride += static_cast<std::size_t>(property.second);
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    throw std::runtime_error("PLY does not contain x/y/z.");
  }

  std::vector<Vec3f> points;
  points.reserve(vertex_count);
  std::vector<char> row(stride);

  for (std::size_t i = 0; i < vertex_count; ++i) {
    input.read(row.data(), static_cast<std::streamsize>(stride));
    if (!input) {
      throw std::runtime_error("Unexpected EOF while reading PLY vertices.");
    }
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    std::memcpy(&x, row.data() + x_offset, sizeof(float));
    std::memcpy(&y, row.data() + y_offset, sizeof(float));
    std::memcpy(&z, row.data() + z_offset, sizeof(float));
    points.push_back({x, y, z});
  }

  return points;
}

std::vector<Region> parseRegions(const std::string & bbox_path)
{
  std::ifstream input(bbox_path);
  if (!input) {
    throw std::runtime_error("Failed to open region bbox file: " + bbox_path);
  }

  std::vector<Region> regions;
  Region current;
  std::string line;

  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    if (line.rfind("Rectangle:", 0) == 0) {
      if (!current.name.empty() && !current.corners.empty()) {
        current.center = makeCenter(current.corners);
        current.bounds_min = makeMinCorner(current.corners);
        current.bounds_max = makeMaxCorner(current.corners);
        current.id = static_cast<int>(regions.size());
        regions.push_back(current);
      }
      current = Region{};
      current.name = line.substr(std::string("Rectangle:").size());
      current.name.erase(0, current.name.find_first_not_of(" \t"));
      continue;
    }

    if (line.find("Corner") != std::string::npos &&
        line.find("X:") != std::string::npos &&
        line.find("Y:") != std::string::npos &&
        line.find("Z:") != std::string::npos)
    {
      std::string cleaned = line;
      std::replace(cleaned.begin(), cleaned.end(), ',', ' ');
      const auto tokens = split(cleaned);
      auto x_it = std::find(tokens.begin(), tokens.end(), "X:");
      auto y_it = std::find(tokens.begin(), tokens.end(), "Y:");
      auto z_it = std::find(tokens.begin(), tokens.end(), "Z:");
      if (x_it == tokens.end() || y_it == tokens.end() || z_it == tokens.end()) {
        continue;
      }
      current.corners.push_back({
        std::stof(*(x_it + 1)),
        std::stof(*(y_it + 1)),
        std::stof(*(z_it + 1))});
    }
  }

  if (!current.name.empty() && !current.corners.empty()) {
    current.center = makeCenter(current.corners);
    current.bounds_min = makeMinCorner(current.corners);
    current.bounds_max = makeMaxCorner(current.corners);
    current.id = static_cast<int>(regions.size());
    regions.push_back(current);
  }

  if (regions.empty()) {
    throw std::runtime_error("No regions parsed from bbox file.");
  }
  return regions;
}

VoxelMap buildVoxelMap(
  const std::vector<Vec3f> & points,
  const std::vector<Region> & regions,
  float resolution,
  float padding)
{
  Vec3f min_corner{
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()};
  Vec3f max_corner{
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest()};

  auto update_bounds = [&](const Vec3f & point) {
      min_corner.x = std::min(min_corner.x, point.x);
      min_corner.y = std::min(min_corner.y, point.y);
      min_corner.z = std::min(min_corner.z, point.z);
      max_corner.x = std::max(max_corner.x, point.x);
      max_corner.y = std::max(max_corner.y, point.y);
      max_corner.z = std::max(max_corner.z, point.z);
    };

  for (const auto & point : points) {
    update_bounds(point);
  }
  for (const auto & region : regions) {
    for (const auto & corner : region.corners) {
      update_bounds(corner);
    }
  }

  min_corner.x -= padding;
  min_corner.y -= padding;
  min_corner.z -= padding;
  max_corner.x += padding;
  max_corner.y += padding;
  max_corner.z += padding;

  const int size_x = static_cast<int>(std::ceil((max_corner.x - min_corner.x) / resolution)) + 1;
  const int size_y = static_cast<int>(std::ceil((max_corner.y - min_corner.y) / resolution)) + 1;
  const int size_z = static_cast<int>(std::ceil((max_corner.z - min_corner.z) / resolution)) + 1;

  VoxelMap map(resolution, min_corner, size_x, size_y, size_z);
  for (const auto & point : points) {
    map.addOccupied(map.worldToGrid(point));
  }
  return map;
}

std::vector<Vec3i> buildInflationOffsets(float inflate_radius, float resolution)
{
  const int radius_voxels = std::max(1, static_cast<int>(std::ceil(inflate_radius / resolution)));
  std::vector<Vec3i> offsets;
  for (int dx = -radius_voxels; dx <= radius_voxels; ++dx) {
    for (int dy = -radius_voxels; dy <= radius_voxels; ++dy) {
      for (int dz = -radius_voxels; dz <= radius_voxels; ++dz) {
        const float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz)) * resolution;
        if (dist <= inflate_radius + kEpsilon) {
          offsets.push_back({dx, dy, dz});
        }
      }
    }
  }
  return offsets;
}

void inflateVoxelMap(VoxelMap & map, float inflate_radius)
{
  const auto offsets = buildInflationOffsets(inflate_radius, map.resolution());
  std::unordered_set<std::int64_t> inflated = map.occupied();
  inflated.reserve(map.occupied().size() * 2U);

  for (const auto & key : map.occupied()) {
    const Vec3i center = map.unpack(key);
    for (const auto & offset : offsets) {
      const Vec3i voxel{center.x + offset.x, center.y + offset.y, center.z + offset.z};
      if (map.inBounds(voxel)) {
        inflated.insert(map.pack(voxel));
      }
    }
  }

  map.setOccupied(std::move(inflated));
}

void buildConvexCover(
  const std::vector<Eigen::Vector3d> & path,
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Vector3d & low_corner,
  const Eigen::Vector3d & high_corner,
  double progress,
  double range,
  gcopter::PolyhedraH & hpolys,
  double eps = 1.0e-6)
{
  hpolys.clear();
  if (path.size() < 2U) {
    return;
  }

  const int n = static_cast<int>(path.size());
  Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
  bd(0, 0) = 1.0;
  bd(1, 0) = -1.0;
  bd(2, 1) = 1.0;
  bd(3, 1) = -1.0;
  bd(4, 2) = 1.0;
  bd(5, 2) = -1.0;

  Eigen::MatrixX4d hp;
  Eigen::MatrixX4d gap;
  Eigen::Vector3d a;
  Eigen::Vector3d b = path.front();
  std::vector<Eigen::Vector3d> valid_points;
  valid_points.reserve(points.size());

  for (int i = 1; i < n;) {
    a = b;
    if ((a - path[static_cast<std::size_t>(i)]).norm() > progress) {
      b = (path[static_cast<std::size_t>(i)] - a).normalized() * progress + a;
    } else {
      b = path[static_cast<std::size_t>(i)];
      ++i;
    }

    bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, high_corner(0));
    bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, low_corner(0));
    bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, high_corner(1));
    bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, low_corner(1));
    bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, high_corner(2));
    bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, low_corner(2));

    valid_points.clear();
    for (const Eigen::Vector3d & point : points) {
      if ((bd.leftCols<3>() * point + bd.rightCols<1>()).maxCoeff() < 0.0) {
        valid_points.push_back(point);
      }
    }

    if (valid_points.empty()) {
      hp = bd;
    } else {
      Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> point_cloud(
        valid_points[0].data(),
        3,
        static_cast<Eigen::Index>(valid_points.size()));
      firi::firi(bd, point_cloud, a, b, hp);
    }

    if (!hpolys.empty()) {
      const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
      const int active_constraints =
        ((hp * ah).array() > -eps).cast<int>().sum() +
        ((hpolys.back() * ah).array() > -eps).cast<int>().sum();
      if (active_constraints >= 3) {
        if (valid_points.empty()) {
          gap = bd;
        } else {
          Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> point_cloud(
            valid_points[0].data(),
            3,
            static_cast<Eigen::Index>(valid_points.size()));
          firi::firi(bd, point_cloud, a, a, gap, 1);
        }
        hpolys.emplace_back(gap);
      }
    }

    hpolys.emplace_back(hp);
  }
}

void shortcutCorridor(gcopter::PolyhedraH & hpolys)
{
  std::vector<Eigen::MatrixX4d> htemp = hpolys;
  if (htemp.empty()) {
    return;
  }
  if (htemp.size() == 1U) {
    htemp.insert(htemp.begin(), htemp.front());
  }

  hpolys.clear();
  const int corridor_size = static_cast<int>(htemp.size());
  bool overlap = false;
  std::deque<int> indices;
  indices.push_front(corridor_size - 1);
  for (int i = corridor_size - 1; i >= 0; --i) {
    for (int j = 0; j < i; ++j) {
      if (j < i - 1) {
        overlap = geo_utils::overlap(htemp[static_cast<std::size_t>(i)], htemp[static_cast<std::size_t>(j)], 0.01);
      } else {
        overlap = true;
      }
      if (overlap) {
        indices.push_front(j);
        i = j + 1;
        break;
      }
    }
  }

  for (const int index : indices) {
    hpolys.push_back(htemp[static_cast<std::size_t>(index)]);
  }
}

std::vector<Vec3i> buildSearchOffsets(float snap_radius, float resolution)
{
  const int max_radius = std::max(1, static_cast<int>(std::ceil(snap_radius / resolution)));
  std::vector<Vec3i> offsets;
  for (int dx = -max_radius; dx <= max_radius; ++dx) {
    for (int dy = -max_radius; dy <= max_radius; ++dy) {
      for (int dz = -max_radius; dz <= max_radius; ++dz) {
        offsets.push_back({dx, dy, dz});
      }
    }
  }
  std::sort(offsets.begin(), offsets.end(), [](const Vec3i & a, const Vec3i & b) {
      const float da = distance3(a, Vec3i{0, 0, 0});
      const float db = distance3(b, Vec3i{0, 0, 0});
      if (std::abs(da - db) > kEpsilon) {
        return da < db;
      }
      return std::abs(a.z) < std::abs(b.z);
    });
  return offsets;
}

Vec3i snapToFreeVoxel(const VoxelMap & map, const Vec3i & seed, const std::vector<Vec3i> & search_offsets)
{
  if (map.inBounds(seed) && !map.isOccupied(seed)) {
    return seed;
  }
  for (const auto & offset : search_offsets) {
    const Vec3i candidate{seed.x + offset.x, seed.y + offset.y, seed.z + offset.z};
    if (map.inBounds(candidate) && !map.isOccupied(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("Failed to snap region center to a free voxel.");
}

bool isWithinSearchBox(
  const Vec3f & point,
  const Region & region,
  float shrink_x,
  float shrink_y,
  float min_search_world_z,
  float max_search_world_z)
{
  const float min_x = std::min(region.center.x, region.bounds_min.x + std::max(0.0F, shrink_x));
  const float max_x = std::max(region.center.x, region.bounds_max.x - std::max(0.0F, shrink_x));
  const float min_y = std::min(region.center.y, region.bounds_min.y + std::max(0.0F, shrink_y));
  const float max_y = std::max(region.center.y, region.bounds_max.y - std::max(0.0F, shrink_y));

  return point.x >= min_x - kEpsilon && point.x <= max_x + kEpsilon &&
         point.y >= min_y - kEpsilon && point.y <= max_y + kEpsilon &&
         point.z >= min_search_world_z - kEpsilon && point.z <= max_search_world_z + kEpsilon;
}

bool isWithinRegionXY(const Vec3f & point, const Region & region)
{
  return point.x >= region.bounds_min.x - kEpsilon && point.x <= region.bounds_max.x + kEpsilon &&
         point.y >= region.bounds_min.y - kEpsilon && point.y <= region.bounds_max.y + kEpsilon;
}

bool isWithinAnyRegionXY(const Vec3f & point, const std::vector<Region> & regions)
{
  for (const auto & region : regions) {
    if (isWithinRegionXY(point, region)) {
      return true;
    }
  }
  return false;
}

bool passesWorldXConstraint(
  const Vec3f & point,
  const std::vector<Region> & regions,
  float min_search_world_x,
  float max_search_world_x)
{
  return ((point.x >= min_search_world_x - kEpsilon && point.x <= max_search_world_x + kEpsilon) ||
         isWithinAnyRegionXY(point, regions));
}

bool isWithinShrunkenMapXY(
  const Vec3f & point,
  const VoxelMap & map,
  float shrink_x,
  float shrink_y)
{
  const float min_x = map.origin().x + std::max(0.0F, shrink_x);
  const float max_x =
    map.origin().x + static_cast<float>(map.sizeX()) * map.resolution() - std::max(0.0F, shrink_x);
  const float min_y = map.origin().y + std::max(0.0F, shrink_y);
  const float max_y =
    map.origin().y + static_cast<float>(map.sizeY()) * map.resolution() - std::max(0.0F, shrink_y);

  return point.x >= min_x - kEpsilon && point.x <= max_x + kEpsilon &&
         point.y >= min_y - kEpsilon && point.y <= max_y + kEpsilon;
}

bool isWithinShrunkenMapXYZ(
  const Vec3f & point,
  const VoxelMap & map,
  const std::vector<Region> & regions,
  float shrink_x,
  float shrink_y,
  float min_search_world_z,
  float max_search_world_z,
  float min_search_world_x,
  float max_search_world_x)
{
  return isWithinShrunkenMapXY(point, map, shrink_x, shrink_y) &&
         passesWorldXConstraint(point, regions, min_search_world_x, max_search_world_x) &&
         point.z >= min_search_world_z - kEpsilon &&
         point.z <= max_search_world_z + kEpsilon;
}

std::optional<Vec3i> findAnchorInRegionSearchBox(
  const VoxelMap & map,
  const Region & region,
  float min_search_world_z,
  float max_search_world_z,
  float shrink_x,
  float shrink_y,
  const std::vector<Vec3i> & search_offsets)
{
  const float min_x = std::min(region.center.x, region.bounds_min.x + std::max(0.0F, shrink_x));
  const float max_x = std::max(region.center.x, region.bounds_max.x - std::max(0.0F, shrink_x));
  const float min_y = std::min(region.center.y, region.bounds_min.y + std::max(0.0F, shrink_y));
  const float max_y = std::max(region.center.y, region.bounds_max.y - std::max(0.0F, shrink_y));

  const int start_x = static_cast<int>(std::floor((min_x - map.origin().x) / map.resolution()));
  const int end_x = static_cast<int>(std::floor((max_x - map.origin().x) / map.resolution()));
  const int start_y = static_cast<int>(std::floor((min_y - map.origin().y) / map.resolution()));
  const int end_y = static_cast<int>(std::floor((max_y - map.origin().y) / map.resolution()));
  const int start_z = static_cast<int>(std::floor((min_search_world_z - map.origin().z) / map.resolution()));
  const int end_z = static_cast<int>(std::floor((max_search_world_z - map.origin().z) / map.resolution()));

  const int clamped_start_x = std::max(0, start_x);
  const int clamped_end_x = std::min(map.sizeX() - 1, std::max(clamped_start_x, end_x));
  const int clamped_start_y = std::max(0, start_y);
  const int clamped_end_y = std::min(map.sizeY() - 1, std::max(clamped_start_y, end_y));
  const int clamped_start_z = std::max(0, start_z);
  const int clamped_end_z = std::min(map.sizeZ() - 1, std::max(clamped_start_z, end_z));

  std::vector<Vec3i> candidate_seeds;
  candidate_seeds.reserve(
    static_cast<std::size_t>(clamped_end_x - clamped_start_x + 1) *
    static_cast<std::size_t>(clamped_end_y - clamped_start_y + 1) *
    static_cast<std::size_t>(clamped_end_z - clamped_start_z + 1));

  const Vec3i center_idx = map.worldToGrid(region.center);
  for (int z = clamped_start_z; z <= clamped_end_z; ++z) {
    for (int y = clamped_start_y; y <= clamped_end_y; ++y) {
      for (int x = clamped_start_x; x <= clamped_end_x; ++x) {
        candidate_seeds.push_back({x, y, z});
      }
    }
  }

  std::sort(candidate_seeds.begin(), candidate_seeds.end(), [&](const Vec3i & a, const Vec3i & b) {
      return distance3(a, center_idx) < distance3(b, center_idx);
    });

  for (const auto & seed : candidate_seeds) {
    if (map.inBounds(seed) &&
        !map.isOccupied(seed) &&
        isWithinSearchBox(map.gridToWorld(seed), region, shrink_x, shrink_y, min_search_world_z, max_search_world_z))
    {
      return seed;
    }
  }

  for (const auto & seed : candidate_seeds) {
    for (const auto & offset : search_offsets) {
      const Vec3i candidate{seed.x + offset.x, seed.y + offset.y, seed.z + offset.z};
      if (!map.inBounds(candidate) || map.isOccupied(candidate)) {
        continue;
      }
      if (!isWithinSearchBox(map.gridToWorld(candidate), region, shrink_x, shrink_y, min_search_world_z, max_search_world_z)) {
        continue;
      }
      return candidate;
    }
  }

  return std::nullopt;
}

bool isVoxelAllowedForSearch(
  const VoxelMap & map,
  const std::vector<Region> & regions,
  const Vec3i & voxel,
  float min_search_world_z,
  float max_search_world_z,
  float min_search_world_x,
  float max_search_world_x,
  float shrink_x,
  float shrink_y)
{
  const Vec3f world = map.gridToWorld(voxel);
  return passesWorldXConstraint(world, regions, min_search_world_x, max_search_world_x) &&
         world.z >= min_search_world_z - kEpsilon &&
         world.z <= max_search_world_z + kEpsilon &&
         isWithinShrunkenMapXY(world, map, shrink_x, shrink_y);
}

Vec3i snapRegionAnchorVoxel(
  const VoxelMap & map,
  const Region & region,
  float min_search_world_z,
  float max_search_world_z,
  float shrink_x,
  float shrink_y,
  const std::vector<Vec3i> & search_offsets)
{
  if (auto strict_anchor = findAnchorInRegionSearchBox(
        map,
        region,
        min_search_world_z,
        max_search_world_z,
        shrink_x,
        shrink_y,
        search_offsets))
  {
    return *strict_anchor;
  }

  if (shrink_x > 0.0F || shrink_y > 0.0F) {
    const auto relaxed_anchor = findAnchorInRegionSearchBox(
      map,
      region,
      min_search_world_z,
      max_search_world_z,
      0.0F,
      0.0F,
      search_offsets);
    if (relaxed_anchor.has_value()) {
      return *relaxed_anchor;
    }
  }

  std::ostringstream oss;
  oss << "Failed to find a free voxel for region " << region.id << " (" << region.name
      << ") within world-z band [" << min_search_world_z << ", " << max_search_world_z
      << "] and shrink_x=" << shrink_x << ", shrink_y=" << shrink_y;
  throw std::runtime_error(oss.str());
}

std::vector<NeighborStep> buildNeighborSteps()
{
  std::vector<NeighborStep> steps;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        const Vec3i delta{dx, dy, dz};
        steps.push_back({delta, distance3(delta, Vec3i{0, 0, 0})});
      }
    }
  }
  std::sort(steps.begin(), steps.end(), [](const NeighborStep & a, const NeighborStep & b) {
      return a.cost < b.cost;
    });
  return steps;
}

DijkstraResult runDijkstraToTargets(
  const VoxelMap & map,
  const std::vector<Region> & regions,
  const Vec3i & start,
  const std::vector<NeighborStep> & neighbors,
  std::size_t max_expansions,
  const std::unordered_set<std::int64_t> & target_keys,
  float min_search_world_z,
  float max_search_world_z,
  float min_search_world_x,
  float max_search_world_x,
  float shrink_x,
  float shrink_y)
{
  DijkstraResult result;
  std::priority_queue<FrontierState, std::vector<FrontierState>, FrontierCompare> frontier;
  std::unordered_set<std::int64_t> closed;

  const auto start_key = map.pack(start);
  result.distance[start_key] = 0.0F;
  frontier.push({start_key, 0.0F});

  std::size_t expansions = 0;
  while (!frontier.empty()) {
    const FrontierState current = frontier.top();
    frontier.pop();

    if (closed.find(current.key) != closed.end()) {
      continue;
    }
    closed.insert(current.key);
    ++expansions;

    if (target_keys.find(current.key) != target_keys.end()) {
      result.reached_targets.insert(current.key);
      if (result.reached_targets.size() == target_keys.size()) {
        break;
      }
    }

    if (expansions > max_expansions) {
      break;
    }

    const Vec3i current_idx = map.unpack(current.key);
    for (const auto & step : neighbors) {
      const Vec3i next{
        current_idx.x + step.delta.x,
        current_idx.y + step.delta.y,
        current_idx.z + step.delta.z};
      if (map.isOccupied(next)) {
        continue;
      }
      if (!isVoxelAllowedForSearch(
            map,
            regions,
            next,
            min_search_world_z,
            max_search_world_z,
            min_search_world_x,
            max_search_world_x,
            shrink_x,
            shrink_y))
      {
        continue;
      }
      const auto next_key = map.pack(next);
      const float tentative_cost = current.cost + step.cost;
      const auto it = result.distance.find(next_key);
      if (it != result.distance.end() && tentative_cost >= it->second) {
        continue;
      }
      result.distance[next_key] = tentative_cost;
      result.parent[next_key] = current.key;
      frontier.push({next_key, tentative_cost});
    }
  }

  return result;
}

std::vector<Vec3i> reconstructPath(
  const DijkstraResult & result,
  const VoxelMap & map,
  std::int64_t start_key,
  std::int64_t goal_key)
{
  std::vector<Vec3i> path;
  auto current = goal_key;
  path.push_back(map.unpack(current));

  while (current != start_key) {
    auto it = result.parent.find(current);
    if (it == result.parent.end()) {
      throw std::runtime_error("Failed to backtrack path from Dijkstra tree.");
    }
    current = it->second;
    path.push_back(map.unpack(current));
  }

  std::reverse(path.begin(), path.end());
  return path;
}

bool lineOfSightIsFree(
  const VoxelMap & map,
  const std::vector<Region> & regions,
  const Vec3i & start,
  const Vec3i & goal,
  float min_search_world_z,
  float max_search_world_z,
  float min_search_world_x,
  float max_search_world_x,
  float shrink_x,
  float shrink_y)
{
  const float dx = static_cast<float>(goal.x - start.x);
  const float dy = static_cast<float>(goal.y - start.y);
  const float dz = static_cast<float>(goal.z - start.z);
  const int steps = std::max(2, static_cast<int>(std::ceil(std::sqrt(dx * dx + dy * dy + dz * dz) * 2.0F)));

  for (int i = 0; i <= steps; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(steps);
    const Vec3i sample{
      static_cast<int>(std::round(start.x + dx * t)),
      static_cast<int>(std::round(start.y + dy * t)),
      static_cast<int>(std::round(start.z + dz * t))};
    if (map.isOccupied(sample)) {
      return false;
    }
    if (!isVoxelAllowedForSearch(
          map,
          regions,
          sample,
          min_search_world_z,
          max_search_world_z,
          min_search_world_x,
          max_search_world_x,
          shrink_x,
          shrink_y))
    {
      return false;
    }
  }
  return true;
}

std::vector<Vec3i> shortenPath(
  const std::vector<Vec3i> & path,
  const VoxelMap & map,
  const std::vector<Region> & regions,
  float min_search_world_z,
  float max_search_world_z,
  float min_search_world_x,
  float max_search_world_x,
  float shrink_x,
  float shrink_y)
{
  if (path.size() <= 2U) {
    return path;
  }

  std::vector<Vec3i> shortened;
  shortened.push_back(path.front());

  std::size_t cursor = 0;
  while (cursor + 1 < path.size()) {
    std::size_t next = path.size() - 1U;
    while (next > cursor + 1U) {
      if (lineOfSightIsFree(
            map,
            regions,
            path[cursor],
            path[next],
            min_search_world_z,
            max_search_world_z,
            min_search_world_x,
            max_search_world_x,
            shrink_x,
            shrink_y))
      {
        break;
      }
      --next;
    }
    shortened.push_back(path[next]);
    cursor = next;
  }
  return shortened;
}

std::vector<Vec3f> resamplePath(const std::vector<Vec3f> & points, float spacing)
{
  if (points.size() <= 1U) {
    return points;
  }

  std::vector<float> cumulative(points.size(), 0.0F);
  for (std::size_t i = 1; i < points.size(); ++i) {
    cumulative[i] = cumulative[i - 1] + distance3(points[i - 1], points[i]);
  }

  const float total_length = cumulative.back();
  if (total_length < kEpsilon) {
    return {points.front()};
  }

  std::vector<Vec3f> samples;
  std::size_t segment = 0;
  for (float dist = 0.0F; dist < total_length; dist += spacing) {
    while (segment + 1U < cumulative.size() && cumulative[segment + 1U] < dist) {
      ++segment;
    }
    if (segment + 1U >= points.size()) {
      samples.push_back(points.back());
      continue;
    }

    const float seg_start = cumulative[segment];
    const float seg_end = cumulative[segment + 1U];
    const float alpha = (dist - seg_start) / std::max(seg_end - seg_start, kEpsilon);
    samples.push_back({
      points[segment].x + alpha * (points[segment + 1U].x - points[segment].x),
      points[segment].y + alpha * (points[segment + 1U].y - points[segment].y),
      points[segment].z + alpha * (points[segment + 1U].z - points[segment].z)});
  }
  samples.push_back(points.back());
  return samples;
}

sensor_msgs::msg::PointCloud2 buildPointCloud2(
  const std::vector<Vec3f> & points,
  const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = frame_id;
  cloud.header.stamp = stamp;
  cloud.height = 1U;
  cloud.width = static_cast<std::uint32_t>(points.size());
  cloud.is_bigendian = false;
  cloud.is_dense = true;
  cloud.point_step = 12U;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.fields.resize(3U);

  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0U;
  cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[0].count = 1U;

  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4U;
  cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[1].count = 1U;

  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8U;
  cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud.fields[2].count = 1U;

  cloud.data.resize(cloud.row_step);
  std::size_t offset = 0U;
  for (const auto & point : points) {
    std::memcpy(cloud.data.data() + offset + 0U, &point.x, sizeof(float));
    std::memcpy(cloud.data.data() + offset + 4U, &point.y, sizeof(float));
    std::memcpy(cloud.data.data() + offset + 8U, &point.z, sizeof(float));
    offset += cloud.point_step;
  }
  return cloud;
}

}  // namespace

class VoxelGuidanceNode : public rclcpp::Node
{
public:
  VoxelGuidanceNode()
  : Node("voxel_guidance_node")
  {
    ply_path_ = declare_parameter<std::string>(
      "ply_path", "/home/zdp/CodeField/tang_swarm_rl/swarm_rl/test/flysite.ply");
    bbox_path_ = declare_parameter<std::string>(
      "bbox_path", "/home/zdp/CodeField/tang_swarm_rl/swarm_rl/test/DR_Surface_BBox_Data.txt");
    output_dir_ = declare_parameter<std::string>(
      "output_dir", "/home/zdp/CodeField/tang_swarm_rl/swarm_rl/test/region_paths_txt");
    output_file_ = declare_parameter<std::string>("output_file", "all_region_pair_paths.txt");
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    resolution_ = declare_parameter<double>("resolution", 0.2);
    inflate_radius_ = declare_parameter<double>("inflate_radius", 0.2);
    padding_ = declare_parameter<double>("padding", 1.0);
    snap_radius_ = declare_parameter<double>("snap_radius", 1.0);
    enable_trajectory_optimization_ =
      declare_parameter<bool>("enable_trajectory_optimization", true);
    trajectory_output_file_ = declare_parameter<std::string>(
      "trajectory_output_file", "all_region_pair_trajectories.json");
    corridor_progress_ = declare_parameter<double>("corridor_progress", 7.0);
    corridor_range_ = declare_parameter<double>("corridor_range", 3.0);
    trajectory_time_weight_ = declare_parameter<double>("trajectory_time_weight", 20.0);
    trajectory_smoothing_eps_ = declare_parameter<double>("trajectory_smoothing_eps", 1.0e-2);
    trajectory_integral_intervals_ =
      std::max<int>(1, declare_parameter<int>("trajectory_integral_intervals", 16));
    trajectory_checkpoint_interval_ =
      std::max<int>(1, declare_parameter<int>("trajectory_checkpoint_interval", 50));
    trajectory_rel_cost_tol_ =
      declare_parameter<double>("trajectory_rel_cost_tol", 1.0e-5);
    trajectory_max_vel_mag_ = declare_parameter<double>("trajectory_max_vel_mag", 1.0);
    trajectory_max_bdr_mag_ = declare_parameter<double>("trajectory_max_bdr_mag", 2.1);
    trajectory_max_tilt_angle_ = declare_parameter<double>("trajectory_max_tilt_angle", 1.05);
    trajectory_min_thrust_ = declare_parameter<double>("trajectory_min_thrust", 2.0);
    trajectory_max_thrust_ = declare_parameter<double>("trajectory_max_thrust", 12.0);
    trajectory_vehicle_mass_ = declare_parameter<double>("trajectory_vehicle_mass", 0.61);
    trajectory_grav_acc_ = declare_parameter<double>("trajectory_grav_acc", 9.8);
    trajectory_horiz_drag_ = declare_parameter<double>("trajectory_horiz_drag", 0.70);
    trajectory_vert_drag_ = declare_parameter<double>("trajectory_vert_drag", 0.80);
    trajectory_paras_drag_ = declare_parameter<double>("trajectory_paras_drag", 0.01);
    trajectory_speed_eps_ = declare_parameter<double>("trajectory_speed_eps", 1.0e-4);
    trajectory_penalty_weights_ = declare_parameter<std::vector<double>>(
      "trajectory_penalty_weights",
      std::vector<double>{1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e5});
    min_search_world_z_ = declare_parameter<double>("min_search_world_z", 1.5);
    max_search_world_z_ = declare_parameter<double>("max_search_world_z", 2.0);
    min_search_world_x_ = declare_parameter<double>("min_search_world_x", -4.0);
    max_search_world_x_ = declare_parameter<double>("max_search_world_x", 12.0);
    map_shrink_x_ = declare_parameter<double>("map_shrink_x", 0.0);
    map_shrink_y_ = declare_parameter<double>("map_shrink_y", 0.0);
    // Backward-compatible aliases from the earlier incorrect interpretation.
    region_shrink_x_ = declare_parameter<double>("region_shrink_x", map_shrink_x_);
    region_shrink_y_ = declare_parameter<double>("region_shrink_y", map_shrink_y_);
    resample_spacing_ = declare_parameter<double>("resample_spacing", 0.25);
    publish_stride_ = std::max<int>(1, declare_parameter<int>("publish_stride", 6));
    max_expansions_ = std::max<int>(1000, declare_parameter<int>("max_expansions", 4000000));

    if (trajectory_penalty_weights_.size() != 5U) {
      throw std::runtime_error("trajectory_penalty_weights must contain exactly 5 entries.");
    }

    voxel_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "voxel_cloud",
      rclcpp::SensorDataQoS());
    region_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "region_markers",
      rclcpp::QoS(1).reliable().transient_local());

    const auto points = loadBinaryPlyPoints(ply_path_);
    surface_points_ = points;
    obstacle_points_eigen_ = toEigenPoints(points);
    regions_ = parseRegions(bbox_path_);

    map_ = std::make_unique<VoxelMap>(
      buildVoxelMap(points, regions_, static_cast<float>(resolution_), static_cast<float>(padding_)));
    inflateVoxelMap(*map_, static_cast<float>(inflate_radius_));

    const auto search_offsets = buildSearchOffsets(static_cast<float>(snap_radius_), static_cast<float>(resolution_));
    for (const auto & region : regions_) {
      snapped_centers_[region.id] = snapRegionAnchorVoxel(
        *map_,
        region,
        static_cast<float>(min_search_world_z_),
        static_cast<float>(max_search_world_z_),
        0.0F,
        0.0F,
        search_offsets);
      const Vec3f anchor_world = map_->gridToWorld(snapped_centers_[region.id]);
      if (!isWithinShrunkenMapXYZ(
            anchor_world,
            *map_,
            regions_,
            effectiveMapShrinkX(),
            effectiveMapShrinkY(),
            static_cast<float>(min_search_world_z_),
            static_cast<float>(max_search_world_z_),
            static_cast<float>(min_search_world_x_),
            static_cast<float>(max_search_world_x_)))
      {
        std::ostringstream oss;
        oss << "Anchor for region " << region.id << " (" << region.name
            << ") falls outside the shrunken map search box.";
        throw std::runtime_error(oss.str());
      }
    }

    computeAndCacheOutputs();
    publishCachedOutputs();
    timer_ = create_wall_timer(
      std::chrono::seconds(2),
      std::bind(&VoxelGuidanceNode::publishCachedOutputs, this));
  }

private:
  float effectiveMapShrinkX() const
  {
    return static_cast<float>(std::max(map_shrink_x_, region_shrink_x_));
  }

  float effectiveMapShrinkY() const
  {
    return static_cast<float>(std::max(map_shrink_y_, region_shrink_y_));
  }

  void computeAndCacheOutputs()
  {
    buildVoxelCloud();
    buildRegionMarkers();
    computeAllPairPaths();
  }

  void buildVoxelCloud()
  {
    std::vector<Vec3f> voxel_centers;
    voxel_centers.reserve(map_->occupied().size() / static_cast<std::size_t>(publish_stride_) + 1U);
    std::size_t count = 0U;
    for (const auto & key : map_->occupied()) {
      if ((count % static_cast<std::size_t>(publish_stride_)) == 0U) {
        voxel_centers.push_back(map_->gridToWorld(map_->unpack(key)));
      }
      ++count;
    }
    voxel_cloud_ = buildPointCloud2(voxel_centers, frame_id_, now());
  }

  void buildRegionMarkers()
  {
    region_markers_.markers.clear();

    visualization_msgs::msg::Marker center_marker;
    center_marker.header.frame_id = frame_id_;
    center_marker.header.stamp = now();
    center_marker.ns = "region_centers";
    center_marker.id = 0;
    center_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    center_marker.action = visualization_msgs::msg::Marker::ADD;
    center_marker.pose.orientation.w = 1.0;
    center_marker.scale.x = resolution_ * 3.0;
    center_marker.scale.y = resolution_ * 3.0;
    center_marker.scale.z = resolution_ * 3.0;

    for (const auto & region : regions_) {
      center_marker.points.push_back(toPoint(region.center));
      center_marker.colors.push_back(makeColor(0.2F, 1.0F, 0.25F, 1.0F));
    }
    region_markers_.markers.push_back(center_marker);

    for (const auto & region : regions_) {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = frame_id_;
      text_marker.header.stamp = now();
      text_marker.ns = "region_labels";
      text_marker.id = 1000 + region.id;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position = toPoint(region.center);
      text_marker.pose.position.z += static_cast<double>(resolution_) * 4.0;
      text_marker.pose.orientation.w = 1.0;
      text_marker.scale.z = resolution_ * 3.0;
      text_marker.color = makeColor(1.0F, 1.0F, 1.0F, 1.0F);
      text_marker.text = std::to_string(region.id) + ": " + region.name;
      region_markers_.markers.push_back(text_marker);
    }
  }

  void computeAllPairPaths()
  {
    pair_paths_.clear();

    const auto neighbors = buildNeighborSteps();
    std::size_t total_pairs = 0;
    std::size_t reachable_pairs = 0;

    for (std::size_t source_index = 0; source_index < regions_.size(); ++source_index) {
      const Region & source_region = regions_[source_index];
      std::unordered_set<std::int64_t> target_keys;
      std::vector<int> target_region_ids;

      for (std::size_t target_index = source_index + 1; target_index < regions_.size(); ++target_index) {
        const Region & target_region = regions_[target_index];
        target_keys.insert(map_->pack(snapped_centers_.at(target_region.id)));
        target_region_ids.push_back(target_region.id);
      }

      if (target_keys.empty()) {
        continue;
      }

      const Vec3i source_idx = snapped_centers_.at(source_region.id);
      const auto source_key = map_->pack(source_idx);

      RCLCPP_INFO(
        get_logger(),
        "Running Dijkstra for source region %d (%s) to %zu remaining targets with world-z band [%.3f, %.3f] and x outside [%.3f, %.3f] filtered only outside region boxes...",
        source_region.id,
        source_region.name.c_str(),
        target_keys.size(),
        min_search_world_z_,
        max_search_world_z_,
        min_search_world_x_,
        max_search_world_x_);

      const auto dijkstra = runDijkstraToTargets(
        *map_,
        regions_,
        source_idx,
        neighbors,
        static_cast<std::size_t>(max_expansions_),
        target_keys,
        static_cast<float>(min_search_world_z_),
        static_cast<float>(max_search_world_z_),
        static_cast<float>(min_search_world_x_),
        static_cast<float>(max_search_world_x_),
        effectiveMapShrinkX(),
        effectiveMapShrinkY());

      RCLCPP_INFO(
        get_logger(),
        "Source region %d reached %zu/%zu targets.",
        source_region.id,
        dijkstra.reached_targets.size(),
        target_keys.size());

      for (int target_region_id : target_region_ids) {
        const Region & target_region = regions_.at(static_cast<std::size_t>(target_region_id));
        const auto goal_key = map_->pack(snapped_centers_.at(target_region.id));

        PathRecord record;
        record.source_id = source_region.id;
        record.source_name = source_region.name;
        record.target_id = target_region.id;
        record.target_name = target_region.name;

        ++total_pairs;
        if (dijkstra.reached_targets.find(goal_key) == dijkstra.reached_targets.end()) {
          record.reachable = false;
          pair_paths_.push_back(std::move(record));
          continue;
        }

        auto voxel_path = reconstructPath(dijkstra, *map_, source_key, goal_key);
        voxel_path = shortenPath(
          voxel_path,
          *map_,
          regions_,
          static_cast<float>(min_search_world_z_),
          static_cast<float>(max_search_world_z_),
          static_cast<float>(min_search_world_x_),
          static_cast<float>(max_search_world_x_),
          effectiveMapShrinkX(),
          effectiveMapShrinkY());
        std::vector<Vec3f> world_path;
        world_path.reserve(voxel_path.size());
        for (const auto & voxel : voxel_path) {
          world_path.push_back(map_->gridToWorld(voxel));
        }
        record.guide_points = world_path;
        record.points = resamplePath(world_path, static_cast<float>(resample_spacing_));
        record.reachable = true;
        pair_paths_.push_back(std::move(record));
        ++reachable_pairs;
      }
    }

    saveAllPathsToText();
    optimizeAndSaveTrajectories();
    RCLCPP_INFO(
      get_logger(),
      "Finished generating %zu/%zu undirected region-pair paths.",
      reachable_pairs,
      total_pairs);
  }

  Eigen::Vector3d mapLowCorner() const
  {
    return Eigen::Vector3d(
      static_cast<double>(map_->origin().x),
      static_cast<double>(map_->origin().y),
      static_cast<double>(map_->origin().z));
  }

  Eigen::Vector3d mapHighCorner() const
  {
    return Eigen::Vector3d(
      static_cast<double>(map_->origin().x) + static_cast<double>(map_->sizeX()) * static_cast<double>(map_->resolution()),
      static_cast<double>(map_->origin().y) + static_cast<double>(map_->sizeY()) * static_cast<double>(map_->resolution()),
      static_cast<double>(map_->origin().z) + static_cast<double>(map_->sizeZ()) * static_cast<double>(map_->resolution()));
  }

  TrajectoryRecord optimizeTrajectoryForPath(const PathRecord & path_record) const
  {
    TrajectoryRecord trajectory_record;
    trajectory_record.source_id = path_record.source_id;
    trajectory_record.source_name = path_record.source_name;
    trajectory_record.target_id = path_record.target_id;
    trajectory_record.target_name = path_record.target_name;
    trajectory_record.path_reachable = path_record.reachable;
    trajectory_record.guide_points = path_record.guide_points;
    trajectory_record.resampled_path = path_record.points;

    if (!path_record.reachable) {
      trajectory_record.failure_reason = "path_unreachable";
      return trajectory_record;
    }
    if (path_record.guide_points.size() < 2U) {
      trajectory_record.failure_reason = "insufficient_guide_points";
      return trajectory_record;
    }
    if (obstacle_points_eigen_.empty()) {
      trajectory_record.failure_reason = "empty_obstacle_cloud";
      return trajectory_record;
    }

    try {
      const std::vector<Eigen::Vector3d> guide_route = toEigenPoints(path_record.guide_points);

      gcopter::PolyhedraH safe_corridor;
      buildConvexCover(
        guide_route,
        obstacle_points_eigen_,
        mapLowCorner(),
        mapHighCorner(),
        corridor_progress_,
        corridor_range_,
        safe_corridor);
      shortcutCorridor(safe_corridor);
      trajectory_record.safe_corridor = exportSafeCorridor(safe_corridor);
      trajectory_record.num_corridor_polyhedra =
        static_cast<int>(trajectory_record.safe_corridor.size());
      if (safe_corridor.empty()) {
        trajectory_record.failure_reason = "empty_safe_corridor";
        return trajectory_record;
      }

      Eigen::Matrix3d initial_pva = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d terminal_pva = Eigen::Matrix3d::Zero();
      initial_pva.col(0) = guide_route.front();
      terminal_pva.col(0) = guide_route.back();

      gcopter::SplineSFCOptimizer optimizer;
      SplineTrajectory::QuinticSpline3D spline;

      Eigen::VectorXd magnitude_bounds(5);
      magnitude_bounds <<
        trajectory_max_vel_mag_,
        trajectory_max_bdr_mag_,
        trajectory_max_tilt_angle_,
        trajectory_min_thrust_,
        trajectory_max_thrust_;

      Eigen::VectorXd penalty_weights(5);
      for (int index = 0; index < 5; ++index) {
        penalty_weights(index) = trajectory_penalty_weights_[static_cast<std::size_t>(index)];
      }

      Eigen::VectorXd physical_params(6);
      physical_params <<
        trajectory_vehicle_mass_,
        trajectory_grav_acc_,
        trajectory_horiz_drag_,
        trajectory_vert_drag_,
        trajectory_paras_drag_,
        trajectory_speed_eps_;

      const bool setup_ok = optimizer.setup(
        trajectory_time_weight_,
        initial_pva,
        terminal_pva,
        safe_corridor,
        std::numeric_limits<double>::infinity(),
        trajectory_smoothing_eps_,
        trajectory_integral_intervals_,
        magnitude_bounds,
        penalty_weights,
        physical_params);
      if (!setup_ok) {
        trajectory_record.failure_reason = "optimizer_setup_failed";
        return trajectory_record;
      }

      const double cost = optimizer.optimize(spline, trajectory_rel_cost_tol_);
      if (!std::isfinite(cost) || !spline.isInitialized() || spline.getNumSegments() <= 0) {
        trajectory_record.failure_reason = "optimizer_failed";
        trajectory_record.optimization_cost = cost;
        return trajectory_record;
      }

      const auto & trajectory = spline.getTrajectory();
      trajectory_record.optimization_succeeded = true;
      trajectory_record.optimization_cost = cost;
      trajectory_record.duration = trajectory.getDuration();
      trajectory_record.num_segments = trajectory.getNumSegments();
      trajectory_record.num_coefficients = trajectory.getNumCoeffs();
      trajectory_record.breakpoints = trajectory.getBreakpoints();
      trajectory_record.coefficients.reserve(static_cast<std::size_t>(trajectory.getCoefficients().rows()));
      for (Eigen::Index row = 0; row < trajectory.getCoefficients().rows(); ++row) {
        trajectory_record.coefficients.push_back({
          static_cast<float>(trajectory.getCoefficients()(row, 0)),
          static_cast<float>(trajectory.getCoefficients()(row, 1)),
          static_cast<float>(trajectory.getCoefficients()(row, 2))});
      }
    } catch (const std::exception & error) {
      trajectory_record.failure_reason = error.what();
    }

    return trajectory_record;
  }

  static void writeVec3ArrayJson(std::ostream & output, const std::vector<Vec3f> & points)
  {
    output << "[";
    for (std::size_t index = 0; index < points.size(); ++index) {
      if (index > 0U) {
        output << ", ";
      }
      output << "["
             << static_cast<double>(points[index].x) << ", "
             << static_cast<double>(points[index].y) << ", "
             << static_cast<double>(points[index].z) << "]";
    }
    output << "]";
  }

  static void writeVec4ArrayJson(std::ostream & output, const std::vector<Vec4d> & values)
  {
    output << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index > 0U) {
        output << ", ";
      }
      output << "["
             << values[index].x << ", "
             << values[index].y << ", "
             << values[index].z << ", "
             << values[index].w << "]";
    }
    output << "]";
  }

  static void writeDoubleArrayJson(std::ostream & output, const std::vector<double> & values)
  {
    output << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index > 0U) {
        output << ", ";
      }
      output << values[index];
    }
    output << "]";
  }

  static std::vector<Vec3f> toVec3fArray(const Eigen::Matrix3Xd & matrix)
  {
    std::vector<Vec3f> values;
    values.reserve(static_cast<std::size_t>(matrix.cols()));
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      values.push_back({
        static_cast<float>(matrix(0, column)),
        static_cast<float>(matrix(1, column)),
        static_cast<float>(matrix(2, column))});
    }
    return values;
  }

  static std::vector<TrajectoryRecord::CorridorPolyhedronRecord> exportSafeCorridor(
    const gcopter::PolyhedraH & safe_corridor)
  {
    std::vector<TrajectoryRecord::CorridorPolyhedronRecord> corridor_records;
    corridor_records.reserve(safe_corridor.size());

    for (const auto & hpoly : safe_corridor) {
      TrajectoryRecord::CorridorPolyhedronRecord record;
      record.halfspaces.reserve(static_cast<std::size_t>(hpoly.rows()));
      for (Eigen::Index row = 0; row < hpoly.rows(); ++row) {
        record.halfspaces.push_back({
          hpoly(row, 0),
          hpoly(row, 1),
          hpoly(row, 2),
          hpoly(row, 3)});
      }

      Eigen::Matrix3Xd vertices;
      if (geo_utils::enumerateVs(hpoly, vertices)) {
        record.vertices = toVec3fArray(vertices);
      }

      corridor_records.push_back(std::move(record));
    }

    return corridor_records;
  }

  void saveAllTrajectoriesToJson() const
  {
    const std::filesystem::path output_dir(output_dir_);
    std::filesystem::create_directories(output_dir);
    const auto output_path = output_dir / trajectory_output_file_;

    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("Failed to open trajectory output file: " + output_path.string());
    }

    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"format_version\": 2,\n";
    output << "  \"export_type\": \"voxel_guidance_gcopter\",\n";
    output << "  \"trajectory_count\": " << trajectory_records_.size() << ",\n";
    output << "  \"path_output_file\": \"" << escapeJsonString(output_file_) << "\",\n";
    output << "  \"trajectory_output_file\": \"" << escapeJsonString(trajectory_output_file_) << "\",\n";
    output << "  \"corridor_progress\": " << corridor_progress_ << ",\n";
    output << "  \"corridor_range\": " << corridor_range_ << ",\n";
    output << "  \"corridor_halfspace_convention\": \"a*x + b*y + c*z + d <= 0\",\n";
    output << "  \"constraints\": {\n";
    output << "    \"MaxVelMag\": " << trajectory_max_vel_mag_ << ",\n";
    output << "    \"MaxBdrMag\": " << trajectory_max_bdr_mag_ << ",\n";
    output << "    \"MaxTiltAngle\": " << trajectory_max_tilt_angle_ << ",\n";
    output << "    \"MinThrust\": " << trajectory_min_thrust_ << ",\n";
    output << "    \"MaxThrust\": " << trajectory_max_thrust_ << "\n";
    output << "  },\n";
    output << "  \"physical_params\": {\n";
    output << "    \"VehicleMass\": " << trajectory_vehicle_mass_ << ",\n";
    output << "    \"GravAcc\": " << trajectory_grav_acc_ << ",\n";
    output << "    \"HorizDrag\": " << trajectory_horiz_drag_ << ",\n";
    output << "    \"VertDrag\": " << trajectory_vert_drag_ << ",\n";
    output << "    \"ParasDrag\": " << trajectory_paras_drag_ << ",\n";
    output << "    \"SpeedEps\": " << trajectory_speed_eps_ << "\n";
    output << "  },\n";
    output << "  \"penalty_weights\": ";
    writeDoubleArrayJson(output, trajectory_penalty_weights_);
    output << ",\n";
    output << "  \"trajectories\": [\n";

    for (std::size_t index = 0; index < trajectory_records_.size(); ++index) {
      const auto & record = trajectory_records_[index];
      output << "    {\n";
      output << "      \"source_id\": " << record.source_id << ",\n";
      output << "      \"source_name\": \"" << escapeJsonString(record.source_name) << "\",\n";
      output << "      \"target_id\": " << record.target_id << ",\n";
      output << "      \"target_name\": \"" << escapeJsonString(record.target_name) << "\",\n";
      output << "      \"path_reachable\": " << (record.path_reachable ? "true" : "false") << ",\n";
      output << "      \"optimization_succeeded\": " << (record.optimization_succeeded ? "true" : "false") << ",\n";
      output << "      \"failure_reason\": \"" << escapeJsonString(record.failure_reason) << "\",\n";
      output << "      \"optimization_cost\": ";
      if (std::isfinite(record.optimization_cost)) {
        output << record.optimization_cost;
      } else {
        output << "null";
      }
      output << ",\n";
      output << "      \"duration\": " << record.duration << ",\n";
      output << "      \"num_segments\": " << record.num_segments << ",\n";
      output << "      \"num_coefficients\": " << record.num_coefficients << ",\n";
      output << "      \"num_corridor_polyhedra\": " << record.num_corridor_polyhedra << ",\n";
      output << "      \"guide_route\": ";
      writeVec3ArrayJson(output, record.guide_points);
      output << ",\n";
      output << "      \"resampled_path\": ";
      writeVec3ArrayJson(output, record.resampled_path);
      output << ",\n";
      output << "      \"safe_corridor\": [\n";
      for (std::size_t corridor_index = 0; corridor_index < record.safe_corridor.size(); ++corridor_index) {
        const auto & polyhedron = record.safe_corridor[corridor_index];
        output << "        {\n";
        output << "          \"halfspaces\": ";
        writeVec4ArrayJson(output, polyhedron.halfspaces);
        output << ",\n";
        output << "          \"vertices\": ";
        writeVec3ArrayJson(output, polyhedron.vertices);
        output << "\n";
        output << "        }";
        if (corridor_index + 1U < record.safe_corridor.size()) {
          output << ",";
        }
        output << "\n";
      }
      output << "      ],\n";
      output << "      \"breakpoints\": ";
      writeDoubleArrayJson(output, record.breakpoints);
      output << ",\n";
      output << "      \"coefficients\": ";
      writeVec3ArrayJson(output, record.coefficients);
      output << "\n";
      output << "    }";
      if (index + 1U < trajectory_records_.size()) {
        output << ",";
      }
      output << "\n";
    }

    output << "  ]\n";
    output << "}\n";
    RCLCPP_INFO(get_logger(), "Saved optimized trajectories to %s", output_path.string().c_str());
  }

  void optimizeAndSaveTrajectories()
  {
    trajectory_records_.clear();
    if (!enable_trajectory_optimization_) {
      RCLCPP_INFO(get_logger(), "Trajectory optimization disabled. Skipping polynomial export.");
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Starting trajectory optimization for %zu region-pair paths. JSON checkpoints will be saved every %d paths.",
      pair_paths_.size(),
      trajectory_checkpoint_interval_);

    trajectory_records_.reserve(pair_paths_.size());
    std::size_t success_count = 0U;
    for (std::size_t index = 0; index < pair_paths_.size(); ++index) {
      const auto & path_record = pair_paths_[index];
      const auto pair_start_time = std::chrono::steady_clock::now();
      RCLCPP_INFO(
        get_logger(),
        "Optimizing trajectory %zu/%zu: %d (%s) -> %d (%s)",
        index + 1U,
        pair_paths_.size(),
        path_record.source_id,
        path_record.source_name.c_str(),
        path_record.target_id,
        path_record.target_name.c_str());

      trajectory_records_.push_back(optimizeTrajectoryForPath(path_record));
      const auto & latest_record = trajectory_records_.back();
      if (latest_record.optimization_succeeded) {
        ++success_count;
      }
      const auto pair_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - pair_start_time).count();

      RCLCPP_INFO(
        get_logger(),
        "Finished trajectory %zu/%zu: success=%s, cost=%s, duration=%.3fs, elapsed=%.3fs",
        index + 1U,
        pair_paths_.size(),
        latest_record.optimization_succeeded ? "true" : "false",
        std::isfinite(latest_record.optimization_cost) ? "finite" : "nonfinite",
        latest_record.duration,
        pair_elapsed);
      if (!latest_record.optimization_succeeded && !latest_record.failure_reason.empty()) {
        RCLCPP_WARN(
          get_logger(),
          "Trajectory %zu/%zu failed: %s",
          index + 1U,
          pair_paths_.size(),
          latest_record.failure_reason.c_str());
      }

      const std::size_t completed = index + 1U;
      const bool should_checkpoint =
        completed == pair_paths_.size() ||
        (completed % static_cast<std::size_t>(trajectory_checkpoint_interval_)) == 0U;
      if (should_checkpoint) {
        saveAllTrajectoriesToJson();
        RCLCPP_INFO(
          get_logger(),
          "Trajectory optimization progress: %zu/%zu completed, %zu succeeded.",
          completed,
          pair_paths_.size(),
          success_count);
      }
    }
    RCLCPP_INFO(
      get_logger(),
      "Optimized %zu/%zu region-pair trajectories.",
      success_count,
      trajectory_records_.size());
  }

  void saveAllPathsToText() const
  {
    const std::filesystem::path output_dir(output_dir_);
    std::filesystem::create_directories(output_dir);
    const auto output_path = output_dir / output_file_;

    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("Failed to open output path file: " + output_path.string());
    }

    output << "num_regions " << regions_.size() << "\n";
    output << "num_pairs " << pair_paths_.size() << "\n";
    output << "resolution " << std::fixed << std::setprecision(3) << resolution_ << "\n";
    output << "inflate_radius " << std::fixed << std::setprecision(3) << inflate_radius_ << "\n";
    output << "min_search_world_z " << std::fixed << std::setprecision(3) << min_search_world_z_ << "\n";
    output << "max_search_world_z " << std::fixed << std::setprecision(3) << max_search_world_z_ << "\n";
    output << "min_search_world_x " << std::fixed << std::setprecision(3) << min_search_world_x_ << "\n";
    output << "max_search_world_x " << std::fixed << std::setprecision(3) << max_search_world_x_ << "\n";
    output << "world_x_rule allow_inside_any_region_xy\n";
    output << "map_shrink_x " << std::fixed << std::setprecision(3) << std::max(map_shrink_x_, region_shrink_x_) << "\n";
    output << "map_shrink_y " << std::fixed << std::setprecision(3) << std::max(map_shrink_y_, region_shrink_y_) << "\n";
    output << "resample_spacing " << std::fixed << std::setprecision(3) << resample_spacing_ << "\n";
    output << "begin_paths\n";

    for (const auto & record : pair_paths_) {
      output << "path "
             << record.source_id << " "
             << record.source_name << " "
             << record.target_id << " "
             << record.target_name;
      if (!record.reachable) {
        output << " unreachable\n";
        continue;
      }
      output << " reachable\n";
      output << "num_points " << record.points.size() << "\n";
      for (const auto & point : record.points) {
        output << std::fixed << std::setprecision(4)
               << point.x << " " << point.y << " " << point.z << "\n";
      }
      output << "end_path\n";
    }

    output << "end_paths\n";
    RCLCPP_INFO(get_logger(), "Saved text path data to %s", output_path.string().c_str());
  }

  void refreshStamps()
  {
    const auto stamp = now();
    voxel_cloud_.header.frame_id = frame_id_;
    voxel_cloud_.header.stamp = stamp;
    for (auto & marker : region_markers_.markers) {
      marker.header.frame_id = frame_id_;
      marker.header.stamp = stamp;
    }
  }

  void publishCachedOutputs()
  {
    refreshStamps();
    voxel_pub_->publish(voxel_cloud_);
    region_pub_->publish(region_markers_);
  }

  std::string ply_path_;
  std::string bbox_path_;
  std::string output_dir_;
  std::string output_file_;
  std::string trajectory_output_file_;
  std::string frame_id_;
  double resolution_{0.1};
  double inflate_radius_{0.25};
  double padding_{1.0};
  double snap_radius_{1.0};
  bool enable_trajectory_optimization_{true};
  double corridor_progress_{7.0};
  double corridor_range_{3.0};
  double trajectory_time_weight_{20.0};
  double trajectory_smoothing_eps_{1.0e-2};
  int trajectory_integral_intervals_{16};
  int trajectory_checkpoint_interval_{50};
  double trajectory_rel_cost_tol_{1.0e-5};
  double trajectory_max_vel_mag_{1.0};
  double trajectory_max_bdr_mag_{2.1};
  double trajectory_max_tilt_angle_{1.05};
  double trajectory_min_thrust_{2.0};
  double trajectory_max_thrust_{12.0};
  double trajectory_vehicle_mass_{0.61};
  double trajectory_grav_acc_{9.8};
  double trajectory_horiz_drag_{0.70};
  double trajectory_vert_drag_{0.80};
  double trajectory_paras_drag_{0.01};
  double trajectory_speed_eps_{1.0e-4};
  std::vector<double> trajectory_penalty_weights_;
  double min_search_world_z_{1.5};
  double max_search_world_z_{2.0};
  double min_search_world_x_{-4.0};
  double max_search_world_x_{12.0};
  double map_shrink_x_{0.0};
  double map_shrink_y_{0.0};
  double region_shrink_x_{0.0};
  double region_shrink_y_{0.0};
  double resample_spacing_{0.25};
  int publish_stride_{6};
  int max_expansions_{4000000};

  std::vector<Vec3f> surface_points_;
  std::vector<Eigen::Vector3d> obstacle_points_eigen_;
  std::vector<Region> regions_;
  std::vector<PathRecord> pair_paths_;
  std::vector<TrajectoryRecord> trajectory_records_;
  std::unordered_map<int, Vec3i> snapped_centers_;
  std::unique_ptr<VoxelMap> map_;

  sensor_msgs::msg::PointCloud2 voxel_cloud_;
  visualization_msgs::msg::MarkerArray region_markers_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr voxel_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr region_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VoxelGuidanceNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
