#pragma once
// by https://github.com/arpaka

#include <fstream>
#include <assert.h>

#include <vector>
#include <string>

namespace katoml {
namespace app {
class MNistLoader {
private:
  std::vector<std::vector<float>> m_images;
  std::vector<int> m_labels;
  int m_size;
  int m_rows;
  int m_cols;

  void load_images(std::string image_file)  {
    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
    char p[4];

    ifs.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 0x803);

    ifs.read(p, 4);
    int size = to_int(p);
    // limit
    if (m_size == 0) m_size = size;
    else m_size = std::min(m_size, size);

    ifs.read(p, 4);
    m_rows = to_int(p);

    ifs.read(p, 4);
    m_cols = to_int(p);

    std::vector<char> q(m_size * m_rows * m_cols);
    ifs.read(q.data(), m_size*m_rows * m_cols);
    m_images.reserve(m_size);
    for (int i=0; i<m_size; ++i) {
      std::vector<float> image(m_rows * m_cols);
      for (int j=0; j<m_rows * m_cols; ++j) {
        image[j] = reinterpret_cast<unsigned char&>(q[j + m_rows*m_cols*i]) / 256.0;
      }
      m_images.push_back(std::move(image));
    }

    ifs.close();
  }

  void load_labels(std::string label_file, int num=0) {
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
    char p[4];

    ifs.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 0x801);

    ifs.read(p, 4);
    int size = to_int(p);
    
    std::vector<char> q(m_size);
    ifs.read(q.data(), m_size);
    m_labels.reserve(m_size);
    for (int i=0; i<m_size; ++i) {
      m_labels.push_back(q[i]);
    }

    ifs.close();
  }

  int to_int(char* p) {
    return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
          ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
  }
public:
  MNistLoader(std::string image_file,
          std::string label_file,
          int num) :
    m_size(num),
    m_rows(0),
    m_cols(0) {
    load_images(image_file);
    load_labels(label_file);
  }

  MNistLoader(std::string image_file,
          std::string label_file) :
    MNistLoader(image_file, label_file, 0) {
  }

  int size() const { return m_size; }
  int rows() const { return m_rows; }
  int cols() const { return m_cols; }

  const std::vector<float>& images(int id) const { return m_images[id]; }
  int labels(int id) { return m_labels[id]; }
};

}
}