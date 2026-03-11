#pragma once
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <cctype>
#include <cmath>

// =============================================================================
// ExprDSL — Compiler + Executor de expressoes com acesso a cols[i+shift]
//
// Suporte: + - * /  >  <  >=  <=  ==  !=  &&  ||  ()  col[shift]  numeros
// Retorno: double (logico: 1.0 = true, 0.0 = false)
//
// Uso correto (performance):
//   auto ast = ExprDSL::compile("close[-1] > open[-1]");  // UMA VEZ
//   double v = ExprDSL::execute(ast, ctx);                // POR BAR
// =============================================================================

namespace ExprDSL {

// -----------------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------------
struct Context {
    const std::map<std::string, std::vector<double>>& data;
    const std::map<std::string, double>&              scalars;
    size_t i;

    double get(const std::string& name, int shift = 0) const {
        auto sc = scalars.find(name);
        if (sc != scalars.end()) return sc->second;
        auto it = data.find(name);
        if (it == data.end()) return 0.0;
        int idx = (int)i + shift;
        if (idx < 0 || idx >= (int)it->second.size()) return 0.0;
        return it->second[idx];
    }
};

// -----------------------------------------------------------------------------
// AST Node — forward declare para NodePtr
// -----------------------------------------------------------------------------
struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node {
    enum class Type {
        Number,    // literal: 1.5
        Scalar,    // nome sem shift  -> scalars lookup, fallback data[i]
        Column,    // nome com shift  -> data[i+shift]
        UnaryNeg,  // -x
        BinOp      // l OP r
    } type = Type::Number;

    double      number = 0.0;
    std::string name;
    int         shift  = 0;
    // op: '+' '-' '*' '/'
    //     '>' '<' 'G'(>=) 'L'(<=) 'E'(==) 'N'(!=)
    //     '&'(&&) '|'(||)
    char        op     = 0;

    NodePtr left;
    NodePtr right;
};

// -----------------------------------------------------------------------------
// execute
// -----------------------------------------------------------------------------
inline double execute(const NodePtr& node, const Context& ctx) {
    if (!node) return 0.0;
    switch (node->type) {
        case Node::Type::Number:
            return node->number;
        case Node::Type::Scalar:
            return ctx.get(node->name, 0);
        case Node::Type::Column:
            return ctx.get(node->name, node->shift);
        case Node::Type::UnaryNeg:
            return -execute(node->left, ctx);
        case Node::Type::BinOp: {
            double l = execute(node->left,  ctx);
            double r = execute(node->right, ctx);
            switch (node->op) {
                case '+': return l + r;
                case '-': return l - r;
                case '*': return l * r;
                case '/': return (r != 0.0) ? l / r : 0.0;
                case '>': return (l >  r) ? 1.0 : 0.0;
                case '<': return (l <  r) ? 1.0 : 0.0;
                case 'G': return (l >= r) ? 1.0 : 0.0;
                case 'L': return (l <= r) ? 1.0 : 0.0;
                case 'E': return (l == r) ? 1.0 : 0.0;
                case 'N': return (l != r) ? 1.0 : 0.0;
                case '&': return (l != 0.0 && r != 0.0) ? 1.0 : 0.0;
                case '|': return (l != 0.0 || r != 0.0) ? 1.0 : 0.0;
                default:  return 0.0;
            }
        }
    }
    return 0.0;
}

// -----------------------------------------------------------------------------
// Compiler — parser recursivo descendente -> AST
// -----------------------------------------------------------------------------
struct Compiler {
    const std::string& src;
    size_t pos;

    explicit Compiler(const std::string& s) : src(s), pos(0) {}

    void skip() {
        while (pos < src.size() && std::isspace((unsigned char)src[pos])) ++pos;
    }

    NodePtr make(Node::Type t) {
        auto n = std::make_shared<Node>();
        n->type = t;
        return n;
    }

    NodePtr compile_expr()  { return compile_or(); }

    NodePtr compile_or() {
        NodePtr left = compile_and();
        skip();
        while (pos + 1 < src.size() && src[pos] == '|' && src[pos+1] == '|') {
            pos += 2;
            NodePtr right = compile_and();
            NodePtr node  = make(Node::Type::BinOp);
            node->op = '|'; node->left = left; node->right = right;
            left = node; skip();
        }
        return left;
    }

    NodePtr compile_and() {
        NodePtr left = compile_cmp();
        skip();
        while (pos + 1 < src.size() && src[pos] == '&' && src[pos+1] == '&') {
            pos += 2;
            NodePtr right = compile_cmp();
            NodePtr node  = make(Node::Type::BinOp);
            node->op = '&'; node->left = left; node->right = right;
            left = node; skip();
        }
        return left;
    }

    NodePtr compile_cmp() {
        NodePtr left = compile_add();
        skip();
        while (pos < src.size()) {
            char op_char = 0;
            if (pos + 1 < src.size() && src[pos+1] == '=') {
                if      (src[pos] == '>') op_char = 'G';
                else if (src[pos] == '<') op_char = 'L';
                else if (src[pos] == '=') op_char = 'E';
                else if (src[pos] == '!') op_char = 'N';
                if (op_char) pos += 2;
            } else if (src[pos] == '>' || src[pos] == '<') {
                op_char = src[pos++];
            }
            if (!op_char) break;
            NodePtr right = compile_add();
            NodePtr node  = make(Node::Type::BinOp);
            node->op = op_char; node->left = left; node->right = right;
            left = node; skip();
        }
        return left;
    }

    NodePtr compile_add() {
        NodePtr left = compile_mul();
        skip();
        while (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) {
            char op = src[pos++];
            NodePtr right = compile_mul();
            NodePtr node  = make(Node::Type::BinOp);
            node->op = op; node->left = left; node->right = right;
            left = node; skip();
        }
        return left;
    }

    NodePtr compile_mul() {
        NodePtr left = compile_unary();
        skip();
        while (pos < src.size() && (src[pos] == '*' || src[pos] == '/')) {
            char op = src[pos++];
            NodePtr right = compile_unary();
            NodePtr node  = make(Node::Type::BinOp);
            node->op = op; node->left = left; node->right = right;
            left = node; skip();
        }
        return left;
    }

    NodePtr compile_unary() {
        skip();
        if (pos < src.size() && src[pos] == '-') {
            ++pos;
            NodePtr node = make(Node::Type::UnaryNeg);
            node->left   = compile_primary();
            return node;
        }
        return compile_primary();
    }

    NodePtr compile_primary() {
        skip();
        if (pos >= src.size()) return make(Node::Type::Number); // 0.0

        // Parenteses
        if (src[pos] == '(') {
            ++pos;
            NodePtr v = compile_expr();
            skip();
            if (pos < src.size() && src[pos] == ')') ++pos;
            return v;
        }

        // Numero literal
        if (std::isdigit((unsigned char)src[pos]) || src[pos] == '.') {
            size_t start = pos;
            while (pos < src.size() &&
                   (std::isdigit((unsigned char)src[pos]) || src[pos] == '.'))
                ++pos;
            NodePtr node = make(Node::Type::Number);
            node->number = std::stod(src.substr(start, pos - start));
            return node;
        }

        // Identificador  +  opcional [shift]
        if (std::isalpha((unsigned char)src[pos]) || src[pos] == '_') {
            size_t start = pos;
            while (pos < src.size() &&
                   (std::isalnum((unsigned char)src[pos]) || src[pos] == '_'))
                ++pos;
            std::string name = src.substr(start, pos - start);

            skip();
            if (pos < src.size() && src[pos] == '[') {
                ++pos; // '['
                bool neg = false;
                if (pos < src.size() && src[pos] == '-') { neg = true; ++pos; }
                size_t ns = pos;
                while (pos < src.size() && std::isdigit((unsigned char)src[pos])) ++pos;
                int shift = 0;
                if (pos > ns)
                    shift = std::stoi(src.substr(ns, pos - ns)) * (neg ? -1 : 1);
                if (pos < src.size() && src[pos] == ']') ++pos;

                NodePtr node = make(Node::Type::Column);
                node->name  = name;
                node->shift = shift;
                return node;
            }

            // Sem shift -> Scalar
            NodePtr node = make(Node::Type::Scalar);
            node->name   = name;
            return node;
        }

        return make(Node::Type::Number); // caracter desconhecido -> 0
    }
};

// -----------------------------------------------------------------------------
// compile: string -> AST (chamar UMA VEZ por expressao)
// -----------------------------------------------------------------------------
inline NodePtr compile(const std::string& expr) {
    if (expr.empty()) return nullptr;
    Compiler c(expr);
    return c.compile_expr();
}

// -----------------------------------------------------------------------------
// evaluate: compatibilidade legado (parse+execute a cada chamada — lento)
// Prefira compile() + execute() no loop.
// -----------------------------------------------------------------------------
inline double evaluate(const std::string& expr, const Context& ctx) {
    if (expr.empty()) return 0.0;
    return execute(compile(expr), ctx);
}

} // namespace ExprDSL