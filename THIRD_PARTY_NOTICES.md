# Third-party notices

VectorFFT is under the MIT license ([`LICENSE`](LICENSE)). The repository also carries files of three other
projects, each under its own license. None of them is part of the library: the library contains no code of
theirs, does not link them, and `cmake --install` installs none of them.

| Project | Version | Files in this repository | Used for | License |
| --- | --- | --- | --- | --- |
| FFTW | 3.3.10 | `gauntlet/fftw/libfftw3-3.dll`, `gauntlet/fftw/libfftw3f-3.dll`, `src/tools/gates/fftw3.dll` | the comparator of the benchmark and test tools, loaded at run time on Windows | GNU GPL, version 2 or later |
| KaTeX | 0.16.21 | its stylesheet and fonts, inside `how-to-use/VectorFFT-Guide.html` and `how-to-use/VectorFFT-Guide.pdf` | the formulas of the User Guide | MIT |
| Computer Modern Unicode fonts | 0.7.0 | inside `how-to-use/VectorFFT-Guide.html`, `how-to-use/VectorFFT-Quick-Guide.html` and `how-to-use/VectorFFT-Guide.pdf` | the text of the User Guide and the Quick Guide | SIL Open Font License 1.1 |

## FFTW

FFTW is copyright (c) 2003, 2007-14 Matteo Frigo and the Massachusetts Institute of Technology, and is free
software under the GNU General Public License, version 2 or later. The license is
[`gauntlet/fftw/COPYING`](gauntlet/fftw/COPYING) and the notice [`gauntlet/fftw/COPYRIGHT`](gauntlet/fftw/COPYRIGHT).
The source of all three DLLs is the FFTW 3.3.10 release, <https://www.fftw.org/fftw-3.3.10.tar.gz>.
[`gauntlet/fftw/README.md`](gauntlet/fftw/README.md) says how the two in `gauntlet/fftw/` are built and which two
build files of the release are changed for it; `src/tools/gates/fftw3.dll` is the FFTW 3.3.10 DLL of the vcpkg
package `fftw3` (x64-windows), unchanged. They are separate programs that ship beside VectorFFT for comparison.

## KaTeX

KaTeX lays out the formulas of the User Guide when the guide is built; the HTML and the PDF carry its stylesheet
and fonts. Its notice:

```
The MIT License (MIT)

Copyright (c) 2013-2020 Khan Academy and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Computer Modern Unicode fonts

The User Guide and the Quick Guide set their text in the Computer Modern Unicode fonts, which the HTML and PDF
files carry. Their notice and license:

```
Copyright (C) Authors of original metafont fonts:
Donald Ervin Knuth (cm, concrete fonts)
1995, 1996, 1997 J"org Knappen, 1990, 1992 Norbert Schwarz (ec fonts)
1992-2006 A.Khodulev, O.Lapko, A.Berdnikov, V.Volovich (lh fonts)
1997-2005 Claudio Beccari (cb greek fonts)
2002 FUKUI Rei (tipa fonts)
2003-2005 Han The Thanh (Vietnamese fonts)
1996-2005 Walter Schmidt (cmbright fonts)

Copyright (C) 2003-2009, Andrey V. Panov (panov@canopus.iacp.dvo.ru),
with Reserved Font Family Name "Computer Modern Unicode fonts".

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
http://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded, 
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```
