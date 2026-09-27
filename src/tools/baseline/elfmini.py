#!/usr/bin/env python3
"""elfmini.py - just enough ELF64 (x86-64, little endian) for obj_equiv's
--strict-data mode: sections, the symbol table and RELA relocations.

No dependency on pyelftools (not installed on the hosts this runs on). Only
relocatable objects (.o) are read; that is all the gate compares.
"""
import struct


class Section:
    __slots__ = ("idx", "name", "type", "flags", "offset", "size", "link",
                 "info", "entsize", "data")


class Symbol:
    __slots__ = ("name", "value", "size", "type", "bind", "shndx")


class Rela:
    __slots__ = ("offset", "sym", "type", "addend")


SHT_SYMTAB, SHT_RELA, SHT_NOBITS = 2, 4, 8
STT_OBJECT, STT_FUNC, STT_SECTION = 1, 2, 3


class Elf:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.raw = f.read()
        raw = self.raw
        if raw[:4] != b"\x7fELF" or raw[4] != 2 or raw[5] != 1:
            raise ValueError("%s: not an ELF64 little-endian object" % path)
        (e_shoff,) = struct.unpack_from("<Q", raw, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", raw, 0x3A)
        self.sections = []
        for i in range(e_shnum):
            o = e_shoff + i * e_shentsize
            (nm, typ, flags, _addr, off, size, link, info, _align,
             entsize) = struct.unpack_from("<IIQQQQIIQQ", raw, o)
            s = Section()
            s.idx, s.type, s.flags, s.offset, s.size = i, typ, flags, off, size
            s.link, s.info, s.entsize = link, info, entsize
            s.name = nm
            s.data = b"" if typ == SHT_NOBITS else raw[off:off + size]
            self.sections.append(s)
        strtab = self.sections[e_shstrndx].data
        for s in self.sections:
            s.name = _cstr(strtab, s.name)
        self.symbols = []
        for s in self.sections:
            if s.type == SHT_SYMTAB:
                names = self.sections[s.link].data
                for k in range(s.size // 24):
                    (nm, info, _other, shndx, value,
                     size) = struct.unpack_from("<IBBHQQ", s.data, k * 24)
                    y = Symbol()
                    y.name = _cstr(names, nm)
                    y.value, y.size, y.shndx = value, size, shndx
                    y.type, y.bind = info & 0xF, info >> 4
                    self.symbols.append(y)
        # relocations, keyed by the section they apply to
        self.relas = {}
        for s in self.sections:
            if s.type == SHT_RELA:
                lst = []
                for k in range(s.size // 24):
                    off, info, addend = struct.unpack_from("<QQq", s.data, k * 24)
                    r = Rela()
                    r.offset, r.sym, r.type, r.addend = off, info >> 32, info & 0xFFFFFFFF, addend
                    lst.append(r)
                self.relas[s.info] = sorted(lst, key=lambda r: r.offset)

    def section_by_name(self, name):
        for s in self.sections:
            if s.name == name:
                return s
        return None


def _cstr(buf, off):
    end = buf.find(b"\0", off)
    return buf[off:end].decode("utf-8", "replace")
