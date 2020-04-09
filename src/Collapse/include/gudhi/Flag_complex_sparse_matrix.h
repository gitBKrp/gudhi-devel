/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Siddharth Pritam
 *
 *    Copyright (C) 2018 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef FLAG_COMPLEX_SPARSE_MATRIX_H_
#define FLAG_COMPLEX_SPARSE_MATRIX_H_

#include <gudhi/Rips_edge_list.h>
#include <gudhi/graph_simplicial_complex.h>

#include <boost/functional/hash.hpp>

#include <Eigen/Sparse>

#ifdef GUDHI_USE_TBB
#include <tbb/parallel_sort.h>
#endif

#include <iostream>
#include <utility>
#include <vector>
#include <queue>
#include <unordered_map>
#include <tuple>
#include <list>
#include <algorithm>
#include <chrono>

#include <ctime>
#include <fstream>

typedef std::size_t Vertex;
using Edge = std::pair<Vertex, Vertex>;  // This is an ordered pair, An edge is stored with convention of the first
                                         // element being the smaller i.e {2,3} not {3,2}. However this is at the level
                                         // of row indices on actual vertex lables
using EdgeFilt = std::pair<Edge, double>;
using edge_list = std::vector<Edge>;

using MapVertexToIndex = std::unordered_map<Vertex, std::size_t>;
using Map = std::unordered_map<Vertex, Vertex>;

using sparseRowMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using rowInnerIterator = sparseRowMatrix::InnerIterator;

using doubleVector = std::vector<double>;
using vertexVector = std::vector<Vertex>;
using boolVector = std::vector<bool>;

using EdgeFiltVector = std::vector<EdgeFilt>;

typedef std::vector<std::tuple<double, Vertex, Vertex>> Filtered_sorted_edge_list;
typedef std::unordered_map<Edge, std::size_t, boost::hash<Edge>> u_edge_to_idx_map;

//!  Class SparseMsMatrix
/*!
  The class for storing the Vertices v/s MaxSimplices Sparse Matrix and performing collapses operations using the N^2()
  Algorithm.
*/
class Flag_complex_sparse_matrix {
 private:
  std::unordered_map<int, Vertex> row_to_vertex;

  // Vertices stored as an unordered_set
  std::unordered_set<Vertex> vertices;

  // Unordered set of  removed edges. (to enforce removal from the matrix)
  std::unordered_set<Edge, boost::hash<Edge>> u_set_removed_redges;

  // Unordered set of  dominated edges. (to inforce removal from the matrix)
  std::unordered_set<Edge, boost::hash<Edge>> u_set_dominated_redges;

  // Map from egde to its index
  u_edge_to_idx_map edge_to_index_map;
  // Boolean vector to indicate if the index is critical or not.
  boolVector critical_edge_indicator;  // critical indicator

  // Boolean vector to indicate if the index is critical or not.
  boolVector dominated_edge_indicator;  // domination indicator

  //! Stores the Map between vertices<B>row_to_vertex  and row indices <B>row_to_vertex -> row-index</B>.
  /*!
    \code
    MapVertexToIndex = std::unordered_map<Vertex,int>
    \endcode
    So, if the original simplex tree had vertices 0,1,4,5 <br>
    <B>row_to_vertex</B> would store : <br>
    \verbatim
    Values =  | 0 | 1 | 4 | 5 |
    Indices =   0   1   2   3
    \endverbatim
    And <B>vertex_to_row</B> would be a map like the following : <br>
    \verbatim
    0 -> 0
    1 -> 1
    4 -> 2
    5 -> 3
    \endverbatim
  */
  MapVertexToIndex vertex_to_row;

  //! Stores the Sparse matrix of double values representing the Original Simplicial Complex.
  /*!
    \code
    sparseRowMatrix   = Eigen::SparseMatrix<double, Eigen::RowMajor> ;
    \endcode
    ;
      */

  sparseRowMatrix sparse_row_adjacency_matrix;  // This is row-major version of the same sparse-matrix, to facilitate easy access
                                       // to elements when traversing the matrix row-wise.

  //! Stores <I>true</I> for dominated rows and  <I>false</I> for undominated rows.
  /*!
    Initialised to a vector of length equal to the value of the variable <B>rows</B> with all <I>false</I> values.
    Subsequent removal of dominated vertices is reflected by concerned entries changing to <I>true</I> in this vector.
  */
  boolVector domination_indicator;  //(domination indicator)

  // Vector of filtered edges, for edge-collapse, the indices of the edges are the row-indices.
  EdgeFiltVector f_edge_vector;

  // Stores the indices from the sorted filtered edge vector.
  // std::set<std::size_t> recurCriticalCoreIndcs;

  //! Stores the number of vertices in the original Simplicial Complex.
  /*!
    This stores the count of vertices (which is also the number of rows in the Matrix).
  */
  std::size_t rows;

  bool edgeCollapsed;

  // Edge e is the actual edge (u,v). Not the row ids in the matrixs
  bool check_edge_domination(Edge e)
  {
    auto u = std::get<0>(e);
    auto v = std::get<1>(e);

    auto rw_u = vertex_to_row[u];
    auto rw_v = vertex_to_row[v];
    auto rw_e = std::make_pair(rw_u, rw_v);
#ifdef DEBUG_TRACES
    std::cout << "The edge {" << u << ", " << v <<  "} is going for domination check." << std::endl;
#endif  // DEBUG_TRACES
    auto commonNeighbours = closed_common_neighbours_row_index(rw_e);
#ifdef DEBUG_TRACES
    std::cout << "And its common neighbours are." << std::endl;
    for (doubleVector::iterator it = commonNeighbours.begin(); it!=commonNeighbours.end(); it++) {
      std::cout << row_to_vertex[*it] << ", " ;
    }
    std::cout<< std::endl;
#endif  // DEBUG_TRACES
    if (commonNeighbours.size() > 2) {
      if (commonNeighbours.size() == 3)
        return true;
      else
        for (doubleVector::iterator it = commonNeighbours.begin(); it != commonNeighbours.end(); it++) {
          auto rw_c = *it;  // Typecasting
          if (rw_c != rw_u and rw_c != rw_v) {
            auto neighbours_c = closed_neighbours_row_index(rw_c);
            // If neighbours_c contains the common neighbours.
            if (std::includes(neighbours_c.begin(), neighbours_c.end(), commonNeighbours.begin(),
                              commonNeighbours.end()))
              return true;
          }
        }
    }
    return false;
  }

  // The edge should be sorted by the indices and indices are original
  bool check_domination_indicator(Edge e)
  {
    return dominated_edge_indicator[edge_to_index_map[e]];
  }

  std::set<std::size_t> three_clique_indices(std::size_t crit) {
    std::set<std::size_t> edge_indices;

    Edge e = std::get<0>(f_edge_vector[crit]);
    Vertex u = std::get<0>(e);
    Vertex v = std::get<1>(e);

#ifdef DEBUG_TRACES
    std::cout << "The  current critical edge to re-check criticality with filt value is : f {" << u << "," << v
              << "} = " << std::get<1>(f_edge_vector[crit]) << std::endl;
#endif  // DEBUG_TRACES
    auto rw_u = vertex_to_row[u];
    auto rw_v = vertex_to_row[v];
    auto rw_critical_edge = std::make_pair(rw_u, rw_v);

    doubleVector commonNeighbours = closed_common_neighbours_row_index(rw_critical_edge);

    if (commonNeighbours.size() > 2) {
      for (doubleVector::iterator it = commonNeighbours.begin(); it != commonNeighbours.end(); it++) {
        auto rw_c = *it;
        if (rw_c != rw_u and rw_c != rw_v) {
          auto e_with_new_nbhr_v = std::minmax(u, row_to_vertex[rw_c]);
          auto e_with_new_nbhr_u = std::minmax(v, row_to_vertex[rw_c]);
          edge_indices.emplace(edge_to_index_map[e_with_new_nbhr_v]);
          edge_indices.emplace(edge_to_index_map[e_with_new_nbhr_u]);
        }
      }
    }
    return edge_indices;
  }

  template<typename FilteredEdgeInsertion>
  void set_edge_critical(std::size_t indx, double filt, FilteredEdgeInsertion filtered_edge_insert) {
#ifdef DEBUG_TRACES
    std::cout << "The curent index  with filtration value " << indx << ", " << filt << " is primary critical" <<
    std::endl;
#endif  // DEBUG_TRACES
    std::set<std::size_t> effectedIndcs = three_clique_indices(indx);
    if (effectedIndcs.size() > 0) {
      for (auto idx = indx - 1; idx > 0; idx--) {
        Edge e = std::get<0>(f_edge_vector[idx]);
        Vertex u = std::get<0>(e);
        Vertex v = std::get<1>(e);
        // If idx is not critical so it should be proceses, otherwise it stays in the graph // prev
        // code : recurCriticalCoreIndcs.find(idx) == recurCriticalCoreIndcs.end()
        if (not critical_edge_indicator[idx]) {
          // If idx is affected
          if (effectedIndcs.find(idx) != effectedIndcs.end()) {
            if (not check_edge_domination(e)) {
#ifdef DEBUG_TRACES
              std::cout << "The curent index became critical " << idx  << std::endl;
#endif  // DEBUG_TRACES
              critical_edge_indicator[idx] = true;
              filtered_edge_insert({u, v}, filt);
              std::set<std::size_t> inner_effected_indcs = three_clique_indices(idx);
              for (auto inr_idx = inner_effected_indcs.rbegin(); inr_idx != inner_effected_indcs.rend(); inr_idx++) {
                if (*inr_idx < idx) effectedIndcs.emplace(*inr_idx);
              }
              inner_effected_indcs.clear();
#ifdef DEBUG_TRACES
              std::cout << "The following edge is critical with filt value: {" << std::get<0>(e) << "," <<
              std::get<1>(e) << "}; " << filt << std::endl;
#endif  // DEBUG_TRACES
            } else
              u_set_dominated_redges.emplace(std::minmax(vertex_to_row[u], vertex_to_row[v]));
          } else
            // Idx is not affected hence dominated.
            u_set_dominated_redges.emplace(std::minmax(vertex_to_row[u], vertex_to_row[v]));
        }
      }
    }
    effectedIndcs.clear();
    u_set_dominated_redges.clear();
  }

  // Returns list of non-zero columns of the particular indx.
  doubleVector closed_neighbours_row_index(double indx)
  {
    doubleVector non_zero_indices;
    Vertex u = indx;
    Vertex v;
#ifdef DEBUG_TRACES
    std::cout << "The neighbours of the vertex: " << row_to_vertex[u] << " are. " << std::endl;
#endif  // DEBUG_TRACES
    if (not domination_indicator[indx]) {
      // Iterate over the non-zero columns
      for (rowInnerIterator it(sparse_row_adjacency_matrix, indx); it; ++it) {
        v = it.index();
        // If the vertex v is not dominated and the edge {u,v} is still in the matrix
        if (not domination_indicator[v] and u_set_removed_redges.find(std::minmax(u, v)) == u_set_removed_redges.end() and
            u_set_dominated_redges.find(std::minmax(u, v)) == u_set_dominated_redges.end()) {
          // inner index, here it is equal to it.columns()
          non_zero_indices.push_back(it.index());
#ifdef DEBUG_TRACES
          std::cout << row_to_vertex[it.index()] << ", " ;
#endif  // DEBUG_TRACES
        }
      }
#ifdef DEBUG_TRACES
      std::cout << std::endl;
#endif  // DEBUG_TRACES
    }
    return non_zero_indices;
  }

  doubleVector closed_common_neighbours_row_index(Edge e)  // Returns the list of closed neighbours of the edge :{u,v}.
  {
    doubleVector common;
    doubleVector non_zero_indices_u;
    doubleVector non_zero_indices_v;
    double u = std::get<0>(e);
    double v = std::get<1>(e);

    non_zero_indices_u = closed_neighbours_row_index(u);
    non_zero_indices_v = closed_neighbours_row_index(v);
    std::set_intersection(non_zero_indices_u.begin(), non_zero_indices_u.end(), non_zero_indices_v.begin(),
                          non_zero_indices_v.end(), std::inserter(common, common.begin()));

    return common;
  }

 public:
  //! Main Constructor
  /*!
    Argument is an instance of Filtered_sorted_edge_list. <br>
    This is THE function that initialises all data members to appropriate values. <br>
    <B>row_to_vertex</B>, <B>vertex_to_row</B>, <B>rows</B>, <B>cols</B>, <B>sparse_row_adjacency_matrix</B> are initialised here.
    <B>domination_indicator</B> are initialised by init() function which is
    called at the begining of this. <br>
  */
  Flag_complex_sparse_matrix(const Filtered_sorted_edge_list& edge_t)
  : rows(0),
    edgeCollapsed(false) {
    for (size_t bgn_idx = 0; bgn_idx < edge_t.size(); bgn_idx++) {
      Vertex u = std::get<1>(edge_t[bgn_idx]);
      Vertex v = std::get<2>(edge_t[bgn_idx]);
      f_edge_vector.push_back({{u, v}, std::get<0>(edge_t[bgn_idx])});
      vertices.emplace(u);
      vertices.emplace(v);
    }
  }

  template<typename OneSkeletonGraph>
  Flag_complex_sparse_matrix(const OneSkeletonGraph& one_skeleton_graph)
  : rows(0),
    edgeCollapsed(false) {
    // Insert all vertices
    for (auto v_it = boost::vertices(one_skeleton_graph); v_it.first != v_it.second; ++v_it.first) {
      vertices.emplace(*(v_it.first));
    }
    // Insert all edges
    for (auto edge_it = boost::edges(one_skeleton_graph);
         edge_it.first != edge_it.second; ++edge_it.first) {
      auto edge = *(edge_it.first);
      Vertex u = source(edge, one_skeleton_graph);
      Vertex v = target(edge, one_skeleton_graph);
      f_edge_vector.push_back({{u, v}, boost::get(Gudhi::edge_filtration_t(), one_skeleton_graph, edge)});
    }
    // Sort edges
    auto sort_by_filtration = [](const EdgeFilt& edge_a, const EdgeFilt& edge_b) -> bool
    {
      return (get<1>(edge_a) < get<1>(edge_b)); 
    };

#ifdef GUDHI_USE_TBB
    tbb::parallel_sort(f_edge_vector.begin(), f_edge_vector.end(), sort_by_filtration);
#else
    std::stable_sort(f_edge_vector.begin(), f_edge_vector.end(), sort_by_filtration);
#endif
  }

  // Performs edge collapse in a decreasing sequence of the filtration value.
  template<typename FilteredEdgeInsertion>
  void filtered_edge_collapse(FilteredEdgeInsertion filtered_edge_insert) {
    std::size_t endIdx = 0;

    u_set_removed_redges.clear();
    u_set_dominated_redges.clear();
    critical_edge_indicator.clear();

    // Initializing sparse_row_adjacency_matrix, This is a row-major sparse matrix.
    sparse_row_adjacency_matrix = sparseRowMatrix(vertices.size(), vertices.size());

    while (endIdx < f_edge_vector.size()) {
      EdgeFilt fec = f_edge_vector[endIdx];
      Edge e = std::get<0>(fec);
      Vertex u = std::get<0>(e);
      Vertex v = std::get<1>(e);
      double filt = std::get<1>(fec);

      // Inserts the edge in the sparse matrix to update the graph (G_i)
      insert_new_edges(u, v, filt);

      edge_to_index_map.emplace(std::minmax(u, v), endIdx);
      critical_edge_indicator.push_back(false);
      dominated_edge_indicator.push_back(false);

      if (not check_edge_domination(e)) {
        critical_edge_indicator[endIdx] = true;
        dominated_edge_indicator[endIdx] = false;
        filtered_edge_insert({u, v}, filt);
        if (endIdx > 1)
          set_edge_critical(endIdx, filt, filtered_edge_insert);
      } else
        dominated_edge_indicator[endIdx] = true;
      endIdx++;
    }

    edgeCollapsed = true;
  }

  void insert_vertex(const Vertex& vertex, double filt_val) {
    auto rw = vertex_to_row.find(vertex);
    if (rw == vertex_to_row.end()) {
      // Initializing the diagonal element of the adjency matrix corresponding to rw_b.
      sparse_row_adjacency_matrix.insert(rows, rows) = filt_val;
      domination_indicator.push_back(false);
      vertex_to_row.insert(std::make_pair(vertex, rows));
      row_to_vertex.insert(std::make_pair(rows, vertex));
      rows++;
    }
  }

  void insert_new_edges(const Vertex& u, const Vertex& v, double filt_val)
  {
    // The edge must not be added before, it should be a new edge.
    insert_vertex(u, filt_val);
    if (u != v) {
      insert_vertex(v, filt_val);
#ifdef DEBUG_TRACES
      std::cout << "Insertion of the edge begins " << u <<", " << v << std::endl;
#endif  // DEBUG_TRACES

      auto rw_u = vertex_to_row.find(u);
      auto rw_v = vertex_to_row.find(v);
#ifdef DEBUG_TRACES
      std::cout << "Inserting the edge " << u <<", " << v << std::endl;
#endif  // DEBUG_TRACES
      sparse_row_adjacency_matrix.insert(rw_u->second, rw_v->second) = filt_val;
      sparse_row_adjacency_matrix.insert(rw_v->second, rw_u->second) = filt_val;
    }
#ifdef DEBUG_TRACES
    else {
     	std::cout << "Already a member simplex,  skipping..." << std::endl;
    }
#endif  // DEBUG_TRACES
  }

  std::size_t num_vertices() const { return vertices.size(); }

};

#endif  // FLAG_COMPLEX_SPARSE_MATRIX_H_
