/**************************************************************************
Copyright: (C) 2021 - 2023 Alexander Shaduri
License: Zlib
***************************************************************************/

#ifndef CSV_PARSER_H
#define CSV_PARSER_H
#if defined(_MSC_VER)
#pragma warning(disable : 4514 5045)
#endif
#include "csv_cell.h"
#include "csv_error.h"
#include "csv_matrix.h"
#include "csv_util.h"

#include <array>
#include <string>
#include <string_view>

namespace Csv {

/*
 * The main CSV parser class.
 *
 * CSV format is defined by RFC 4180:
 * https://www.ietf.org/rfc/rfc4180.txt
 *
 * RFC 4180 CSV format summary:
 * - Each line ends with CRLF.
 * - CRLF is optional for the last line.
 * - There is an optional header line (in the same format as the rest of the
 * data).
 * - Each line must have the same number of fields. Spaces are part of the
 * cells. There is no trailing comma on the line.
 * - A field _may_ be enclosed in double-quotes. If the field is not enclosed in
 * double-quotes, there may not be a double quote inside.
 * - Fields containing double-quotes, commas, and newlines _must_ be enclosed in
 * double-quotes.
 * - Double-quotes are escaped by repeating them, like "".
 *
 * Considerations:
 * - There may be implementations using other line breaks than CRLF.
 * - A parser should be liberal in accepting variations of CSV format.
 *
 * Our parser implementation details:
 * - Cell types are determined automatically. There is a String, Double, and
 * Empty cell type.
 * - Numeric cell must be unquoted to be treated as Double.
 * - Numeric cells may be preceded and/or followed by a whitespace (space or
 * tab) (for easy loading of hand-typed numeric data).
 * - Standard C++, Matlab, and a few implementation-specific floating point
 * formats are accepted as doubles.
 * - A cell is treated as Empty type only if it's completely empty and unquoted;
 * calling parser.useEmptyCellType(false) disables the Empty type completely.
 * - If a quoted cell is preceded and/or followed by whitespace (space or tab),
 * this whitespace is ignored.
 * - A cell is treated as Double only if it's unquoted and can be completely
 * parsed as a string representation of a double.
 * - Escaped quotes inside unquoted strings are supported, but only if they are
 * not at the beginning of a cell (ignoring the whitespace).
 * - DOS, UNIX, Mac line endings are supported; Excel on Mac uses/used Mac
 * endings for CSV.
 * - Line ending format inside strings is preserved.
 * - getOriginalStringView() methods may return escaped double-quotes;
 * string_views are read-only, and we cannot touch the original CSV data; use
 * getCleanString() methods if you need unescaped data.
 */
class Parser {
public:
  /// If set to true, empty cell type is a separate type from (empty) string.
  /// Default: true.
  inline constexpr void useEmptyCellType(bool use_empty_cell_type);

  /// Check whether an empty cell type is a separate type from (empty) string.
  [[nodiscard]] inline constexpr bool useEmptyCellType() const;

  /// Parse CSV string data and store the results using a callback function.
  /// The calling order is to always pass the first row, then the second row,
  /// etc... Callback function signature: void func(std::size_t row, std::size_t
  /// column, std::string_view cell_data, CellTypeHint hint). \throws ParseError
  template <typename StoreCellFunction>
  constexpr void parse(std::string_view data,
                       StoreCellFunction storeCellFunc) const;

  /// Parse CSV string data into a vector of columns. The innermost type may be
  /// one of Cell* classes
  // (e.g., CellReference), or a primitive numeric type (e.g., double).
  /// \param data CSV string data.
  /// \param[out] values An empty 2D vector to store the data in. Accepts types
  /// like std::vector<std::vector<CellReference>>. \throws ParseError
  template <typename Vector2D>
  constexpr void parseTo2DVector(std::string_view data, Vector2D &values) const;

  /// Parse CSV string to 2D std::array, an array of columns.
  /// This method conveniently wraps parse() function to simplify compile-time
  /// parsing. \param data CSV string data. \tparam rows Number of rows \tparam
  /// columns Number of columns \tparam Cell Type of cell in array, e.g.
  /// CellStringReference \return std::array<std::array<Cell, rows>, columns>
  /// \throws ParseError
  template <std::size_t rows, std::size_t columns,
            typename Cell = CellStringReference>
  constexpr auto parseTo2DArray(std::string_view data) const;

  /// Parse CSV string data into a flat matrix in row-major format (A11, A12,
  /// A13, A21, ...). The number of rows and columns is determined
  /// automatically. \param data string data to parse. \param[out] values a flat
  /// (empty) matrix. This will be resized automatically. Accepts types like
  /// std::vector<double>. \return matrix information \throws ParseError
  template <typename Vector>
  constexpr MatrixInformation parseToVectorRowMajor(std::string_view data,
                                                    Vector &values) const;

  /// Parse CSV string data into a flat matrix in row-major format (A11, A12,
  /// A13, A21, ...). \param data string data to parse. \param[out] values a
  /// flat (empty) matrix. This will be resized automatically. Accepts types
  /// like std::vector<double>. \param rows_hint std::nullopt, or the number of
  /// rows which will help with optimizing allocations. \param columns The
  /// number of columns. If std::nullopt, the number of columns is determined
  /// automatically. \return matrix information \throws ParseError
  template <typename Vector>
  constexpr MatrixInformation
  parseToVectorRowMajor(std::string_view data, Vector &values,
                        std::optional<std::size_t> rows_hint,
                        std::optional<std::size_t> columns) const;

  /// Parse CSV string data into a flat matrix in column-major format (A11, A21,
  /// A31, A12, ...). \param data string data to parse. \param[out] values a
  /// flat (empty) matrix. This will be resized automatically. Accepts types
  /// like std::vector<double>. \param rows The number of rows, required. It is
  /// needed to calculate offsets in output container. \param columns_hint
  /// std::nullopt, or the number of columns which will help with optimizing
  /// allocations. \return matrix information \throws ParseError
  template <typename Vector>
  constexpr MatrixInformation
  parseToVectorColumnMajor(std::string_view data, Vector &values,
                           std::size_t rows,
                           std::optional<std::size_t> columns_hint) const;

  /// Parse CSV string to 1D std::array, stored in row-major or column-major
  /// order. This method conveniently wraps parse() function to simplify
  /// compile-time parsing. \param data CSV string data. \tparam rows Number of
  /// rows \tparam columns Number of columns \tparam Cell Type of cell in array,
  /// e.g. CellStringReference \return std::array<Cell, rows*columns> \throws
  /// ParseError
  template <std::size_t rows, std::size_t columns,
            typename Cell = CellStringReference>
  constexpr auto parseToArray(std::string_view data, MatrixOrder order) const;

private:
  /// Parser state machine state
  enum class MachineState {
    AtCellStart,
    InLeadingWhiteSpace,
    InsideUnquotedValue,
    InsideQuotedValue,
    AfterQuotedValue,
  };

  /// Internal state maintained during parsing
  struct ParserState {
    MachineState machine_state = MachineState::AtCellStart;
    std::size_t current_row = 0, current_column = 0;
    std::string_view current_value;
    bool escaped_quotes_encountered = false;

    /// Switch to next column
    constexpr void switchToNextColumn() {
      current_value = std::string_view();
      escaped_quotes_encountered = false;
      ++current_column;
    }

    /// Switch to next line.
    /// \return new current position
    [[nodiscard]] constexpr std::size_t
    switchToNextLine(const std::string_view &data, std::size_t current_pos) {
      current_value = std::string_view();
      escaped_quotes_encountered = false;
      current_column = 0;
      ++current_row;
      // If it's CR, and the next character is LF, skip LF as well.
      if (auto next_char = peek(data, current_pos);
          data[current_pos] == '\r' && next_char == '\n') {
        ++current_pos;
      }
      return current_pos;
    }

    /// Advance current value's end point by \ref by_chars
    constexpr void increaseCurrentValueSize(std::size_t by_chars) {
      current_value = std::string_view(current_value.data(),
                                       current_value.size() + by_chars);
    }

    /// Reset current value to specific start point and size
    constexpr void restartCurrentValue(const std::string_view &data,
                                       std::size_t current_pos,
                                       std::size_t size) {
      current_value = data.substr(current_pos, size);
      escaped_quotes_encountered = false;
    }
  };

  /// Read a character from \ref data at position (current_pos + advance_chars).
  /// \return std::char_traits<char>::eof() if position is past the data's
  /// contents.
  [[nodiscard]] static constexpr std::char_traits<char>::int_type
  peek(const std::string_view &data, std::size_t current_pos,
       std::size_t advance_chars = 1) {
    if ((current_pos + advance_chars) < data.size()) {
      return std::char_traits<char>::to_int_type(
          data[current_pos + advance_chars]);
    }
    return std::char_traits<char>::eof();
  }

  /// Read a character from \ref data at position \ref pos.
  /// \return std::char_traits<char>::eof() if position is past the data's
  /// contents.
  [[nodiscard]] static constexpr std::char_traits<char>::int_type
  readChar(const std::string_view &data, std::size_t pos) {
    return peek(data, pos, 0);
  }

  /// Store current value using a callback function.
  /// StoreCellFunction is of the same type as in parse().
  template <typename StoreCellFunction>
  constexpr void store(StoreCellFunction storeCellFunc,
                       const ParserState &state, CellTypeHint type_hint) const {
    if (type_hint == CellTypeHint::Empty && !use_empty_cell_type_) {
      type_hint = CellTypeHint::StringWithoutEscapedQuotes;
    }
    storeCellFunc(state.current_row, state.current_column, state.current_value,
                  type_hint);
  }

  /// If set to true, empty cell type is a separate type from (empty) string.
  bool use_empty_cell_type_ = true;
};

// ---- Implementation

constexpr void Parser::useEmptyCellType(bool use_empty_cell_type) {
  use_empty_cell_type_ = use_empty_cell_type;
}

constexpr bool Parser::useEmptyCellType() const { return use_empty_cell_type_; }

template <typename StoreCellFunction>
constexpr void Parser::parse(std::string_view data,
                             StoreCellFunction storeCellFunc) const {
  ParserState state;

  for (std::size_t pos = 0; pos <= data.size(); ++pos) {
    auto current_char = readChar(data, pos);

    switch (state.machine_state) {
    // Starting the cell
    case MachineState::AtCellStart: {
      switch (current_char) {
      case ' ':
      case '\t':
        // Store the whitespace
        state.machine_state = MachineState::InLeadingWhiteSpace;
        state.restartCurrentValue(data, pos, 1);
        break;
      case '\"':
        // Start a quoted cell.
        // Discard the starting quote as well, but provide a "Quoted" hint when
        // the quoted cell is stored.
        state.machine_state = MachineState::InsideQuotedValue;
        state.restartCurrentValue(data, pos + 1, 0);
        break;
      case ',':
        // Empty cell. Store the value.
        store(storeCellFunc, state, CellTypeHint::Empty);
        state.switchToNextColumn();
        break;
      case '\r':
      case '\n':
        // Empty cell (trailing comma; last value on the line). Store the value.
        store(storeCellFunc, state, CellTypeHint::Empty);
        state.machine_state = MachineState::AtCellStart;
        // Handle CRLF if needed and set the state to the next line, cell start.
        pos = state.switchToNextLine(data, pos);
        break;
      case std::char_traits<char>::eof():
        // If it's in the first column, it's a trailing newline, nothing else to
        // do. Otherwise, it's a last empty cell on the line after comma (aka
        // trailing comma).
        if (state.current_column != 0) {
          store(storeCellFunc, state, CellTypeHint::Empty);
        }
        return;
      default:
        // Start an unquoted cell
        state.machine_state = MachineState::InsideUnquotedValue;
        state.restartCurrentValue(data, pos, 1);
        break;
      }
      break;
    }

    // Only whitespace encountered in the cell so far
    case MachineState::InLeadingWhiteSpace: {
      switch (current_char) {
      case ' ':
      case '\t':
        // More whitespace. Append to existing leading whitespace.
        state.increaseCurrentValueSize(1);
        break;
      case '\"':
        // Quote encountered. Discard the leading whitespace, start a quoted
        // cell. Discard the starting quote as well, but provide a "Quoted" hint
        // when the quoted cell is stored.
        state.machine_state = MachineState::InsideQuotedValue;
        state.restartCurrentValue(data, pos + 1, 0);
        break;
      case ',':
        // Whitespace-only string cell. Store the value.
        store(storeCellFunc, state, CellTypeHint::StringWithoutEscapedQuotes);
        state.machine_state = MachineState::AtCellStart;
        state.switchToNextColumn();
        break;
      case '\r':
      case '\n':
        // Whitespace-only string cell (last value on the line). Store the
        // value.
        store(storeCellFunc, state, CellTypeHint::StringWithoutEscapedQuotes);
        state.machine_state = MachineState::AtCellStart;
        // Handle CRLF if needed and set the state to the next line, cell start.
        pos = state.switchToNextLine(data, pos);
        break;
      case std::char_traits<char>::eof():
        // Store the value, exit.
        store(storeCellFunc, state, CellTypeHint::StringWithoutEscapedQuotes);
        return;
      default:
        // Continue an unquoted cell
        state.machine_state = MachineState::InsideUnquotedValue;
        state.increaseCurrentValueSize(1);
        break;
      }
      break;
    }

    // We encountered non-whitespace characters in a cell, and it didn't start
    // with a quote.
    case MachineState::InsideUnquotedValue: {
      switch (current_char) {
      case '\"':
        // Make sure the next character is also a quote, otherwise it's a format
        // error. We don't accept unescaped double-quotes in unquoted strings
        // because it leads to ambiguity.
        if (peek(data, pos) != '\"') {
          throw ParseError(state.current_row, state.current_column);
        }
        // Continue unquoted string, consume the second quote as well.
        ++pos;
        state.increaseCurrentValueSize(2);
        state.escaped_quotes_encountered =
            true; // used for hints when storing the value
        break;
      case ',':
        // End of cell. Store the value.
        store(storeCellFunc, state,
              state.escaped_quotes_encountered
                  ? CellTypeHint::StringWithEscapedQuotes
                  : CellTypeHint::UnquotedData);
        state.machine_state = MachineState::AtCellStart;
        state.switchToNextColumn();
        break;
      case '\r':
      case '\n':
        // End of line. Store the value.
        store(storeCellFunc, state,
              state.escaped_quotes_encountered
                  ? CellTypeHint::StringWithEscapedQuotes
                  : CellTypeHint::UnquotedData);
        state.machine_state = MachineState::AtCellStart;
        // Handle CRLF if needed and set the state to the next line, cell start.
        pos = state.switchToNextLine(data, pos);
        break;
      case std::char_traits<char>::eof():
        // Store the value, exit.
        store(storeCellFunc, state,
              state.escaped_quotes_encountered
                  ? CellTypeHint::StringWithEscapedQuotes
                  : CellTypeHint::UnquotedData);
        return;
      default:
        // Continue an unquoted cell
        state.increaseCurrentValueSize(1);
        break;
      }
      break;
    }

    // The cell started with an optional whitespace and a quote, we're past the
    // first quote.
    case MachineState::InsideQuotedValue: {
      switch (current_char) {
      case '\"':
        // If the next character is also a quote, it's an escaped quote.
        if (peek(data, pos) == '\"') {
          // Continue quoted string, consume the second quote as well.
          ++pos;
          state.increaseCurrentValueSize(2);
          state.escaped_quotes_encountered =
              true; // used for hints when storing the value
        } else {
          // End of quoted value. Store the value, discard the ending quote.
          store(storeCellFunc, state,
                state.escaped_quotes_encountered
                    ? CellTypeHint::StringWithEscapedQuotes
                    : CellTypeHint::StringWithoutEscapedQuotes);
          state.machine_state = MachineState::AfterQuotedValue;
        }
        break;
      case std::char_traits<char>::eof():
        // EOF while inside a quoted cell, throw an error.
        throw ParseError(state.current_row, state.current_column);
      default:
        // Continue an unquoted cell
        state.increaseCurrentValueSize(1);
        break;
      }
      break;
    }

    // The quoted cell just ended
    case MachineState::AfterQuotedValue: {
      switch (current_char) {
      case ' ':
      case '\t':
        // Just whitespace, ignore it.
        break;
      case ',':
        // End of cell. The value has been stored already, switch to the next
        // line.
        state.machine_state = MachineState::AtCellStart;
        state.switchToNextColumn();
        break;
      case '\r':
      case '\n':
        // End of line. The value has been stored already, switch to the next
        // line.
        state.machine_state = MachineState::AtCellStart;
        pos = state.switchToNextLine(data, pos);
        break;
      case std::char_traits<char>::eof():
        // Nothing more to do, return.
        return;
      default:
        // Anything else is an error
        throw ParseError(state.current_row, state.current_column);
      }
      break;
    }
    }
  }
}

template <typename Vector2D>
constexpr void Parser::parseTo2DVector(std::string_view data,
                                       Vector2D &values) const {
  Vector2D parsed_values;
  parse(data, [&parsed_values](std::size_t row, std::size_t column,
                               std::string_view cell_data, CellTypeHint hint) {
    if (parsed_values.size() < (column + 1)) {
      parsed_values.resize(column + 1);
    }
    if (parsed_values[column].size() < (row + 1)) {
      parsed_values[column].resize(row + 1);
    }
    parsed_values[column][row] =
        CellTrait<typename Vector2D::value_type::value_type>::create(cell_data,
                                                                     hint);
  });
  std::swap(values, parsed_values);
}

template <std::size_t rows, std::size_t columns, typename Cell>
constexpr auto Parser::parseTo2DArray(std::string_view data) const {
  std::array<std::array<Cell, rows>, columns> matrix;

  parse(data, [&matrix](std::size_t row, std::size_t column,
                        std::string_view cell_data,
                        Csv::CellTypeHint hint) constexpr {
    matrix[column][row] = CellTrait<Cell>::create(cell_data, hint);
  });

  return matrix;
}

template <typename Vector>
constexpr MatrixInformation
Parser::parseToVectorRowMajor(std::string_view data, Vector &values) const {
  return parseToVectorRowMajor(data, values, std::nullopt, std::nullopt);
}

template <typename Vector>
constexpr MatrixInformation
Parser::parseToVectorRowMajor(std::string_view data, Vector &values,
                              std::optional<std::size_t> rows_hint,
                              std::optional<std::size_t> columns) const {
  Vector parsed_values;
  if (rows_hint.has_value() && columns.has_value()) {
    parsed_values.reserve(rows_hint.value() * columns.value());
  }

  MatrixInformation info;
  info.setOrder(MatrixOrder::RowMajor);

  bool unknown_columns = !columns.has_value();
  if (columns.has_value()) {
    info.setColumns(columns.value());
  }
  std::size_t read_columns = 0;

  // The lambda is called in row-major order, so we can exploit that to
  // determine the number of columns.
  parse(data,
        [&](std::size_t row, std::size_t column, std::string_view cell_data,
            [[maybe_unused]] CellTypeHint hint) {
          if (unknown_columns) {
            if (row == 0) {
              read_columns = column + 1;
              parsed_values.resize(read_columns);
            } else if (row == 1) {
              unknown_columns = false;
              info.setColumns(read_columns);
              if (rows_hint.has_value()) {
                parsed_values.reserve(rows_hint.value() * info.getColumns());
              }
            }
          }

          if (!unknown_columns) {
            if (parsed_values.size() < ((row + 1) * info.getColumns())) {
              parsed_values.resize((row + 1) * info.getColumns());
            }
          }
          parsed_values[info.matrixIndex(row, column)] =
              CellTrait<typename Vector::value_type>::create(cell_data, hint);
        });
  std::swap(values, parsed_values);

  if (values.empty()) {
    MatrixInformation empty_info;
    empty_info.setOrder(MatrixOrder::RowMajor);
    return empty_info;
  }

  info.setRows(info.getColumns() == 0 ? 0
                                      : (values.size() / info.getColumns()));

  return info;
}

template <typename Vector>
constexpr MatrixInformation Parser::parseToVectorColumnMajor(
    std::string_view data, Vector &values, std::size_t rows,
    std::optional<std::size_t> columns_hint) const {
  Vector parsed_values;
  if (columns_hint.has_value()) {
    parsed_values.reserve(columns_hint.value() * rows);
  }

  MatrixInformation info;
  info.setOrder(MatrixOrder::ColumnMajor);
  info.setRows(rows);

  parse(data,
        [&](std::size_t row, std::size_t column, std::string_view cell_data,
            [[maybe_unused]] CellTypeHint hint) {
          if (parsed_values.size() < ((column + 1) * rows)) {
            parsed_values.resize((column + 1) * rows);
          }
          parsed_values[info.matrixIndex(row, column)] =
              CellTrait<typename Vector::value_type>::create(cell_data, hint);
        });
  std::swap(values, parsed_values);

  if (values.empty()) {
    MatrixInformation empty_info;
    empty_info.setOrder(MatrixOrder::ColumnMajor);
    return empty_info;
  }

  info.setColumns(info.getRows() == 0 ? 0 : (values.size() / info.getRows()));

  return info;
}

template <std::size_t rows, std::size_t columns, typename Cell>
constexpr auto Parser::parseToArray(std::string_view data,
                                    MatrixOrder order) const {
  std::array<Cell, rows * columns> matrix;

  parse(data, [&matrix, order](std::size_t row, std::size_t column,
                               std::string_view cell_data,
                               Csv::CellTypeHint hint) constexpr {
    matrix[MatrixInformation::matrixIndex(row, column, rows, columns, order)] =
        CellTrait<Cell>::create(cell_data, hint);
  });

  return matrix;
}

} // namespace Csv

#endif
