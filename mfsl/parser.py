import logging

import ply.lex as lex
import ply.yacc as yacc


class Parser(object):
    tokens = (
        "BOOL", "KEYWORD", "FLOAT", "INTEGER",
        "LPAREN", "RPAREN", "LITERAL",
        "COMMA", "EQUAL",
    )

    def t_BOOL(self, t):
        r"(true|false)"
        t.value = t.value == "true"
        return t

    t_KEYWORD = r"[a-zA-Z][\w\-\.]*"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_COMMA = ","
    t_EQUAL = "="
    t_ignore = " \t\r\n"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lexer = self.build_lexer()
        self.yacc = self.build_yacc()

    def t_LITERAL(self, t):
        r"""
        ('[^']*'|"[^"]*")
        """
        t.value = t.value[1:-1]
        return t

    def t_FLOAT(self, t):
        r"\-?\d+\.\d*"
        t.value = float(t.value)
        return t

    def t_INTEGER(self, t):
        r"\-?\d+"
        t.value = int(t.value)
        return t

    def t_error(self, t):
        self.logger.error(f"unknown character met by lexer: {t}")

    def p_feature(self, p):
        """
        feature : KEYWORD LPAREN arguments RPAREN
                | KEYWORD LPAREN arguments COMMA RPAREN
                | KEYWORD LPAREN RPAREN
        """
        args = dict()
        if len(p) >= 5:
            args = p[3]
        p[0] = {
            "feat": p[1],
            "args": args.get("args", tuple()),
            "kwargs": args.get("kwargs", dict())
        }

    def p_value(self, p):
        """
        value : INTEGER
              | KEYWORD
              | FLOAT
              | LITERAL
              | feature
              | BOOL
        """
        valtype = p.slice[1].type.lower()
        p[0] = {
            "type": valtype,
            "value": p[1]
        }

    def p_keyvalue(self, p):
        """
        keyvalue : KEYWORD EQUAL value
        """
        p[0] = (p[1], p[3])

    def p_arguments(self, p):
        """
        arguments : pos_arguments
                  | keyval_arguments
                  | pos_arguments COMMA keyval_arguments
        """
        args = dict()
        if len(p) == 2:
            argtype = p.slice[1].type
            if argtype == "pos_arguments":
                args["args"] = p[1]
            elif argtype == "keyval_arguments":
                args["kwargs"] = p[1]
        else:
            args["args"] = p[1]
            args["kwargs"] = p[3]
        p[0] = args

    def p_pos_arguments(self, p):
        """
        pos_arguments : pos_arguments COMMA value
                      | value
        """
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 4:
            p[0] = p[1]
            p[0].append(p[3])

    def p_keyval_arguments(self, p):
        """
        keyval_arguments : keyval_arguments COMMA keyvalue
                         | keyvalue
        """
        if len(p) == 2:
            p[0] = {p[1][0]: p[1][1]}
        elif len(p) == 4:
            p[0] = p[1]
            p[0][p[3][0]] = p[3][1]

    def p_error(self, p):
        self.logger.error(f"parsing error encountered by yacc: {p}")
        raise ValueError()

    def build_lexer(self, **kwargs):
        return lex.lex(module=self, **kwargs)

    def build_yacc(self, **kwargs):
        return yacc.yacc(module=self, start="feature", **kwargs)

    def parse(self, text):
        try:
            return self.yacc.parse(text, lexer=self.lexer)
        except ValueError:
            return
