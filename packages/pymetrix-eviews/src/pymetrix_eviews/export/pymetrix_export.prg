' pymetrix_export.prg -- writes the open EViews workfile as a pymetrix bundle (format version 1)
'
' EXPERIMENTAL. UNTESTED IN EVIEWS.
' This program was written from EViews's public documentation, without EViews: the pymetrix project has no EViews
' licence, so it has never been run. Every line marked (verify) uses syntax or behaviour that must be checked
' against EViews's own Command and Programming Reference, in the two most recent major versions, before the
' "experimental" label comes off. Until then, save the workfile from EViews as CSV or Excel (File > Save As, or
' wfsave(type=...)) and load that in pymetrix: that is the supported way in.
'
' Use: open the workfile in EViews, then run
'     exec pymetrix_export.prg C:\work\macro-export
' and open C:\work\macro-export\<workfile>.evbundle in pymetrix (File > Import EViews workfile...).
'
' What it writes (see the plugin's README for the format):
'     <workfile>.evbundle          the manifest (JSON), written last: a folder without it is an incomplete export
'     pages\<page>.csv             each page's series and alpha series, with a date column (pmx_date)
'     text\<page>\<object>.txt     each model's text
'     objects\<page>\<name>.csv    vectors, matrices, sym and coef objects; each equation's estimates
'     valmaps\<page>\<name>.csv    value maps: value,label
'     export.log                   one line per object: exported, or skipped and why
'
' It changes nothing in the workfile: its own objects are named _pmx_* and deleted at the end, and each page's
' sample is put back as it was.
'
' Program version 1.0. pymetrix-eviews notes when a bundle comes from an older program than the one it ships.

' ---------------------------------------------------------------------------------------------- settings
' User attributes to export for every series (EViews has no function that lists them (verify)): names, space
' separated, e.g. %pmx_attrs = "seasadj base_year"
%pmx_attrs = ""

' ---------------------------------------------------------------------------------------------- the folder
%pmx_out = %0    ' the program's first argument (exec prog arg): standard
if %pmx_out = "" then
    seterr "give the folder to write to: exec pymetrix_export.prg C:\work\macro-export"   ' (verify: seterr)
    stop
endif
if @right(%pmx_out, 1) = "\" then
    %pmx_out = @left(%pmx_out, @len(%pmx_out) - 1)
endif
shell(h) mkdir "{%pmx_out}"   ' (verify: shell(h) runs a command without a window; mkdir of an existing folder only warns)
shell(h) mkdir "{%pmx_out}\pages"
shell(h) mkdir "{%pmx_out}\text"
shell(h) mkdir "{%pmx_out}\objects"
shell(h) mkdir "{%pmx_out}\valmaps"

%pmx_q = @chr(34)   ' a double quote (verify: @chr)
%pmx_bs = "\"
%pmx_wf = @lower(@wfname)   ' (verify: @wfname)
%pmx_wfpath = @wfpath       ' (verify: @wfpath gives the folder the workfile was opened from)
%pmx_start_page = @pagename ' (verify: @pagename)

text _pmx_json   ' the manifest, a line at a time (verify: text objects and .append)
text _pmx_log

' ---------------------------------------------------------------------------------------------- JSON strings
' %pmx_s in, %pmx_j out: the text as a JSON string, with its quotes. Backslash first, then the double quote, then
' the control characters a label could hold. (verify: global subroutines see the program's % variables)
subroutine pmx_json(string %pmx_in)
    %pmx_t = @replace(%pmx_in, %pmx_bs, %pmx_bs + %pmx_bs)   ' (verify: @replace replaces every occurrence)
    %pmx_t = @replace(%pmx_t, %pmx_q, %pmx_bs + %pmx_q)
    %pmx_t = @replace(%pmx_t, @chr(13), %pmx_bs + "r")
    %pmx_t = @replace(%pmx_t, @chr(10), %pmx_bs + "n")
    %pmx_t = @replace(%pmx_t, @chr(9), %pmx_bs + "t")
    %pmx_j = %pmx_q + %pmx_t + %pmx_q
endsub

' a number as JSON: full precision, NA as null (verify: the "g.17" format keeps 17 significant digits)
subroutine pmx_num(scalar !pmx_x)
    if @isna(!pmx_x) then
        %pmx_n = "null"
    else
        %pmx_n = @str(!pmx_x, "g.17")
    endif
endsub

' ---------------------------------------------------------------------------------------------- the header
call pmx_json(@ver)   ' (verify: @ver gives the EViews version as text)
%pmx_ver = %pmx_j
call pmx_json(@strnow("YYYY-MM-DD") + "T" + @strnow("HH:MI:SS"))   ' (verify: @strnow and its format letters)
%pmx_now = %pmx_j
call pmx_json(%pmx_wf)
%pmx_wfj = %pmx_j
call pmx_json(%pmx_wfpath)
%pmx_wfpj = %pmx_j
call pmx_json(@lower(%pmx_start_page))
%pmx_apj = %pmx_j

_pmx_json.append {
_pmx_json.append   "format": "pymetrix-eviews-bundle",
_pmx_json.append   "format_version": 1,
_pmx_json.append   "exporter": {"program_version": "1.0", "eviews_version": {%pmx_ver}, "exported": {%pmx_now}},
_pmx_json.append   "workfile": {"name": {%pmx_wfj}, "path": {%pmx_wfpj}, "active_page": {%pmx_apj}},
_pmx_json.append   "na": "NA",
_pmx_json.append   "pages": [
' (verify: .append writes the rest of the line as it is, braces and quotes included; if it does not, build each
' line in a % variable and append that)

' ---------------------------------------------------------------------------------------------- page by page
%pmx_pages = @pagelist   ' (verify: @pagelist, the workfile's pages separated by spaces)
for !pmx_p = 1 to @wcount(%pmx_pages)
    %pmx_pg = @word(%pmx_pages, !pmx_p)
    %pmx_pgl = @lower(%pmx_pg)
    pageselect {%pmx_pg}
    %pmx_oldsmpl = @pagesmpl   ' the page's current sample, put back below (verify: @pagesmpl)
    smpl @all
    shell(h) mkdir "{%pmx_out}\text\{%pmx_pgl}"
    shell(h) mkdir "{%pmx_out}\objects\{%pmx_pgl}"
    shell(h) mkdir "{%pmx_out}\valmaps\{%pmx_pgl}"

    ' the structure and frequency: "U" undated; otherwise A, Q, M, W, D5, D7, ... as EViews names them
    %pmx_freq = @pagefreq   ' (verify: @pagefreq, and the codes it gives for weekly and the daily frequencies)
    if %pmx_freq = "U" then
        %pmx_struct = "undated"
    else
        %pmx_struct = "dated"
    endif
    if @ispanel then   ' (verify: @ispanel)
        %pmx_struct = "panel"
    endif

    ' the date and id columns, made here and deleted at the end of the page
    %pmx_datecol = ""
    %pmx_ids = ""
    if %pmx_struct <> "undated" then
        %pmx_datecol = "pmx_date"
        if @isobject(%pmx_datecol) then
            %pmx_datecol = "pmx_date_"   ' the user's own object of that name is left alone
        endif
        alpha {%pmx_datecol} = @datestr(@date, "YYYY-MM-DD")   ' the first day of each period (verify: @date, @datestr's format)
    endif
    if %pmx_struct = "panel" then
        %pmx_ids = "pmx_crossid"
        series {%pmx_ids} = @crossid   ' the cross-section number (verify: @crossid)
    endif
    if %pmx_struct = "undated" then
        %pmx_ids = "pmx_obs"
        series {%pmx_ids} = @obsnum   ' (verify: @obsnum)
    endif

    ' the series and alpha series, page CSV first
    %pmx_ser = @wlookup("*", "series")   ' (verify: @wlookup's type names)
    %pmx_alp = @wlookup("*", "alpha")
    %pmx_cols = %pmx_datecol + " " + %pmx_ids
    for !pmx_i = 1 to @wcount(%pmx_ser)
        %pmx_o = @word(%pmx_ser, !pmx_i)
        if @lower(%pmx_o) <> @lower(%pmx_ids) then
            %pmx_cols = %pmx_cols + " " + %pmx_o
        endif
    next
    for !pmx_i = 1 to @wcount(%pmx_alp)
        %pmx_o = @word(%pmx_alp, !pmx_i)
        if @lower(%pmx_o) <> @lower(%pmx_datecol) then
            %pmx_cols = %pmx_cols + " " + %pmx_o
        endif
    next
    ' (verify: pagesave's text options: the delimiter, the NA string, numbers at full precision, alpha values in
    ' quotes, one header row of names, no extra date column of EViews's own, and @keep of series and alpha series only;
    ' and that link and frml series are written with their values)
    pagesave(type=text, delim=",", na="NA") "{%pmx_out}\pages\{%pmx_pgl}.csv" @keep {%pmx_cols} @smpl @all
    _pmx_log.append {%pmx_pg}: page written to pages\{%pmx_pgl}.csv

    call pmx_json(%pmx_pgl)
    %pmx_pgj = %pmx_j
    call pmx_json(%pmx_oldsmpl)
    %pmx_smj = %pmx_j
    call pmx_json(@otod(1))   ' (verify: @otod gives the date of an observation, as EViews writes dates)
    %pmx_r1 = %pmx_j
    call pmx_json(@otod(@obsrange))   ' (verify: @obsrange)
    %pmx_r2 = %pmx_j
    if !pmx_p > 1 then
        _pmx_json.append ,
    endif
    _pmx_json.append {
    _pmx_json.append "name": {%pmx_pgj}, "structure": "{%pmx_struct}", "frequency": "{%pmx_freq}",
    _pmx_json.append "range": [{%pmx_r1}, {%pmx_r2}], "sample": {%pmx_smj}, "file": "pages/{%pmx_pgl}.csv",
    if %pmx_datecol = "" then
        _pmx_json.append "date_column": null,
    else
        _pmx_json.append "date_column": "{%pmx_datecol}",
    endif
    if %pmx_ids = "" then
        _pmx_json.append "id_columns": [],
    else
        _pmx_json.append "id_columns": ["{%pmx_ids}"],
    endif

    ' each series: its kind and descriptive members
    _pmx_json.append "series": [
    !pmx_first = 1
    for !pmx_k = 1 to 2
        if !pmx_k = 1 then
            %pmx_list = %pmx_ser
            %pmx_kind = "numeric"
        else
            %pmx_list = %pmx_alp
            %pmx_kind = "alpha"
        endif
        for !pmx_i = 1 to @wcount(%pmx_list)
            %pmx_o = @word(%pmx_list, !pmx_i)
            if @lower(%pmx_o) = @lower(%pmx_datecol) or @lower(%pmx_o) = @lower(%pmx_ids) then
                next   ' (verify: "next" inside an if skips to the next pass; if not, wrap the rest in an else)
            endif
            call pmx_json(@lower(%pmx_o))
            %pmx_line = "{""name"": " + %pmx_j + ", ""kind"": ""{%pmx_kind}"""
            ' (verify: "" inside a string literal is a double quote; if not, build these with %pmx_q)
            call pmx_json({%pmx_o}.@displayname)   ' (verify: the data members @displayname, @description, @units,
            %pmx_line = %pmx_line + ", ""display_name"": " + %pmx_j   ' @source and @remarks, and in which versions)
            call pmx_json({%pmx_o}.@description)
            %pmx_line = %pmx_line + ", ""description"": " + %pmx_j
            call pmx_json({%pmx_o}.@units)
            %pmx_line = %pmx_line + ", ""units"": " + %pmx_j
            call pmx_json({%pmx_o}.@source)
            %pmx_line = %pmx_line + ", ""source"": " + %pmx_j
            call pmx_json({%pmx_o}.@remarks)
            %pmx_line = %pmx_line + ", ""remarks"": " + %pmx_j
            %pmx_line = %pmx_line + ", ""attributes"": {"
            for !pmx_a = 1 to @wcount(%pmx_attrs)
                %pmx_an = @word(%pmx_attrs, !pmx_a)
                call pmx_json({%pmx_o}.@attr(%pmx_an))   ' (verify: .@attr("name"))
                %pmx_aj = %pmx_j
                call pmx_json(%pmx_an)
                if !pmx_a > 1 then
                    %pmx_line = %pmx_line + ", "
                endif
                %pmx_line = %pmx_line + %pmx_j + ": " + %pmx_aj
            next
            %pmx_line = %pmx_line + "}"
            %pmx_vm = {%pmx_o}.@map   ' the value map attached to the series, if any (verify: how to read it)
            if %pmx_vm = "" then
                %pmx_line = %pmx_line + ", ""value_map"": null}"
            else
                call pmx_json(@lower(%pmx_vm))
                %pmx_line = %pmx_line + ", ""value_map"": " + %pmx_j + "}"
            endif
            if !pmx_first = 0 then
                %pmx_line = ", " + %pmx_line
            endif
            !pmx_first = 0
            _pmx_json.append {%pmx_line}
            _pmx_log.append {%pmx_pg}\{%pmx_o}: {%pmx_kind} series exported
        next
    next
    _pmx_json.append ],

    ' groups: their members (verify: @members)
    _pmx_json.append "groups": [
    %pmx_list = @wlookup("*", "group")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        call pmx_json(@lower(%pmx_o))
        %pmx_line = "{""name"": " + %pmx_j + ", ""members"": ["
        %pmx_mem = {%pmx_o}.@members
        for !pmx_m = 1 to @wcount(%pmx_mem)
            call pmx_json(@lower(@word(%pmx_mem, !pmx_m)))
            if !pmx_m > 1 then
                %pmx_line = %pmx_line + ", "
            endif
            %pmx_line = %pmx_line + %pmx_j
        next
        %pmx_line = %pmx_line + "]}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: group exported
    next
    _pmx_json.append ],

    ' samples: their text (verify: how a sample object gives its text; @smpl is assumed)
    _pmx_json.append "samples": [
    %pmx_list = @wlookup("*", "sample")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        call pmx_json(@lower(%pmx_o))
        %pmx_line = "{""name"": " + %pmx_j
        call pmx_json({%pmx_o}.@smpl)
        %pmx_line = %pmx_line + ", ""text"": " + %pmx_j + "}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: sample exported
    next
    _pmx_json.append ],

    ' scalars and strings, in the manifest
    _pmx_json.append "scalars": [
    %pmx_list = @wlookup("*", "scalar")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        call pmx_json(@lower(%pmx_o))
        call pmx_num({%pmx_o})
        %pmx_line = "{""name"": " + %pmx_j + ", ""value"": " + %pmx_n + "}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: scalar exported
    next
    _pmx_json.append ],
    _pmx_json.append "strings": [
    %pmx_list = @wlookup("*", "string")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        call pmx_json(@lower(%pmx_o))
        %pmx_line = "{""name"": " + %pmx_j
        call pmx_json({%pmx_o})
        %pmx_line = %pmx_line + ", ""value"": " + %pmx_j + "}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: string exported
    next
    _pmx_json.append ],

    ' vectors, matrices, sym and coef: a CSV each, one line per row, NA for missing (written cell by cell, so the
    ' format is the program's own and not EViews's display)
    _pmx_json.append "arrays": [
    !pmx_first = 1
    %pmx_types = "vector matrix sym coef"
    for !pmx_k = 1 to 4
        %pmx_ty = @word(%pmx_types, !pmx_k)
        %pmx_list = @wlookup("*", %pmx_ty)
        for !pmx_i = 1 to @wcount(%pmx_list)
            %pmx_o = @word(%pmx_list, !pmx_i)
            %pmx_ol = @lower(%pmx_o)
            !pmx_r = @rows({%pmx_o})       ' (verify: @rows and @columns of each of the four types)
            !pmx_c = @columns({%pmx_o})
            text _pmx_arr
            for !pmx_y = 1 to !pmx_r
                %pmx_row = ""
                for !pmx_x = 1 to !pmx_c
                    call pmx_num({%pmx_o}(!pmx_y, !pmx_x))   ' (verify: (i, j) indexing of a vector and a coef)
                    if %pmx_n = "null" then
                        %pmx_n = "NA"
                    endif
                    if !pmx_x > 1 then
                        %pmx_row = %pmx_row + ","
                    endif
                    %pmx_row = %pmx_row + %pmx_n
                next
                _pmx_arr.append {%pmx_row}
            next
            _pmx_arr.save "{%pmx_out}\objects\{%pmx_pgl}\{%pmx_ol}.csv"   ' (verify: text.save writes the lines as they are)
            delete _pmx_arr
            call pmx_json(%pmx_ol)
            %pmx_line = "{""name"": " + %pmx_j + ", ""type"": ""{%pmx_ty}"", ""rows"": " + @str(!pmx_r) + ", ""cols"": " + @str(!pmx_c)
            %pmx_line = %pmx_line + ", ""file"": ""objects/{%pmx_pgl}/{%pmx_ol}.csv""}"
            if !pmx_first = 0 then
                %pmx_line = ", " + %pmx_line
            endif
            !pmx_first = 0
            _pmx_json.append {%pmx_line}
            _pmx_log.append {%pmx_pg}\{%pmx_o}: {%pmx_ty} exported
        next
    next
    _pmx_json.append ],

    ' value maps: value,label (verify: freezing a valmap's sheet into a table and saving it as CSV gives these two
    ' columns with that header; if not, write them cell by cell as the arrays are)
    _pmx_json.append "value_maps": [
    %pmx_list = @wlookup("*", "valmap")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        %pmx_ol = @lower(%pmx_o)
        freeze(_pmx_vt) {%pmx_o}.sheet
        _pmx_vt.save(t=csv) "{%pmx_out}\valmaps\{%pmx_pgl}\{%pmx_ol}.csv"
        delete _pmx_vt
        call pmx_json(%pmx_ol)
        %pmx_line = "{""name"": " + %pmx_j + ", ""file"": ""valmaps/{%pmx_pgl}/{%pmx_ol}.csv""}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: value map exported
    next
    _pmx_json.append ],

    ' equations: the command, method, sample and EViews's estimates (kept to compare, not imported as results)
    _pmx_json.append "equations": [
    %pmx_list = @wlookup("*", "equation")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        %pmx_ol = @lower(%pmx_o)
        !pmx_nc = {%pmx_o}.@ncoef
        text _pmx_est
        _pmx_est.append coefficient,estimate,std_error
        for !pmx_y = 1 to !pmx_nc
            call pmx_num({%pmx_o}.@coefs(!pmx_y))
            %pmx_b = %pmx_n
            call pmx_num({%pmx_o}.@stderrs(!pmx_y))
            _pmx_est.append c({!pmx_y}),{%pmx_b},{%pmx_n}
        next
        _pmx_est.save "{%pmx_out}\objects\{%pmx_pgl}\{%pmx_ol}_est.csv"
        delete _pmx_est
        call pmx_json(%pmx_ol)
        %pmx_line = "{""name"": " + %pmx_j
        call pmx_json({%pmx_o}.@command)   ' (verify: @command; a command over several lines would go to text\)
        %pmx_line = %pmx_line + ", ""command"": " + %pmx_j
        call pmx_json({%pmx_o}.@method)    ' (verify: @method)
        %pmx_line = %pmx_line + ", ""method"": " + %pmx_j
        call pmx_json({%pmx_o}.@smpl)      ' (verify: an equation's @smpl is its estimation sample)
        %pmx_line = %pmx_line + ", ""sample"": " + %pmx_j + ", ""ncoef"": " + @str(!pmx_nc)
        %pmx_line = %pmx_line + ", ""estimates"": ""objects/{%pmx_pgl}/{%pmx_ol}_est.csv""}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: equation exported
    next
    _pmx_json.append ],

    ' models: the text, as the Text view shows it (verify: freezing the text view and saving the frozen text)
    _pmx_json.append "models": [
    %pmx_list = @wlookup("*", "model")
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        %pmx_ol = @lower(%pmx_o)
        freeze(_pmx_mt) {%pmx_o}.text
        _pmx_mt.save "{%pmx_out}\text\{%pmx_pgl}\{%pmx_ol}.txt"
        delete _pmx_mt
        call pmx_json(%pmx_ol)
        %pmx_line = "{""name"": " + %pmx_j + ", ""text"": ""text/{%pmx_pgl}/{%pmx_ol}.txt""}"
        if !pmx_i > 1 then
            %pmx_line = ", " + %pmx_line
        endif
        _pmx_json.append {%pmx_line}
        _pmx_log.append {%pmx_pg}\{%pmx_o}: model exported
    next
    _pmx_json.append ],

    ' everything else: listed with its type (verify: @wlookup("*") lists every object, and .@type gives its type)
    _pmx_json.append "skipped": [
    %pmx_done = "series alpha group sample scalar string vector matrix sym coef valmap equation model"
    %pmx_list = @wlookup("*")
    !pmx_first = 1
    for !pmx_i = 1 to @wcount(%pmx_list)
        %pmx_o = @word(%pmx_list, !pmx_i)
        %pmx_ty = @lower({%pmx_o}.@type)
        if @wfind(%pmx_done, %pmx_ty) = 0 and @left(@lower(%pmx_o), 5) <> "_pmx_" then   ' (verify: @wfind)
            call pmx_json(@lower(%pmx_o))
            %pmx_line = "{""name"": " + %pmx_j + ", ""type"": ""{%pmx_ty}"", ""reason"": ""objects of this type are not exported""}"
            if !pmx_first = 0 then
                %pmx_line = ", " + %pmx_line
            endif
            !pmx_first = 0
            _pmx_json.append {%pmx_line}
            _pmx_log.append {%pmx_pg}\{%pmx_o}: skipped ({%pmx_ty} objects are not exported)
        endif
    next
    _pmx_json.append ]
    _pmx_json.append }

    ' the page as it was
    if %pmx_datecol <> "" then
        delete {%pmx_datecol}
    endif
    if %pmx_ids <> "" then
        delete {%pmx_ids}
    endif
    smpl {%pmx_oldsmpl}
next

_pmx_json.append ]
_pmx_json.append }

' ---------------------------------------------------------------------------------------------- the end
pageselect {%pmx_start_page}
_pmx_log.save "{%pmx_out}\export.log"
' the manifest last: a folder without it is an incomplete export, and pymetrix refuses it
' (verify: the encoding text.save writes; pymetrix reads UTF-8 and falls back to Windows-1252 with a note)
_pmx_json.save "{%pmx_out}\{%pmx_wf}.evbundle"
delete _pmx_json _pmx_log
statusline pymetrix bundle written to {%pmx_out}\{%pmx_wf}.evbundle
