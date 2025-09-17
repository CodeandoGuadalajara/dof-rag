
-- DOF Headers Filter: Convert Word styles to Markdown headers and clean unwanted attributes

local style_map = {
  ["CABEZA"] = 1,
  ["Titulo 1"] = 2,
  ["Titulo 2"] = 3,
  ["ANOTACION"] = 4,
}

local text_styles = {
  "texto", "texto1", "text", "normal", "párrafo", "parrafo", "paragraph",
  "romanos", "Plain Text", "Hyperlink", "footnote text", "INCISO",
  "Footnote Characters", "Notas al pie", "Estilo", "footnote reference", "Normal (Web)"
}

local function get_custom_style(attr)
  return attr and attr.attributes and 
    (attr.attributes["custom-style"] or attr.attributes["custom-style-name"] or attr.attributes["style"])
end

local function is_text_style(style)
  if not style then return false end
  local style_lower = style:lower()
  for _, text_style in ipairs(text_styles) do
    if style_lower == text_style:lower() then
      return true
    end
  end
  return false
end

function Div(el)
  local style = get_custom_style(el.attr)
  
  if is_text_style(style) then
    return el.content
  end
  
  local level = style and style_map[style]
  if not level then return nil end

  local new_blocks = {}
  local header_created = false

  for _, blk in ipairs(el.content) do
    if not header_created and (blk.t == "Para" or blk.t == "Plain") then
      table.insert(new_blocks, pandoc.Header(level, blk.content))
      header_created = true
    else
      table.insert(new_blocks, blk)
    end
  end

  if header_created then
    return new_blocks
  else
    local txt = pandoc.utils.stringify(el)
    if txt ~= "" then
      return { pandoc.Header(level, { pandoc.Str(txt) }) }
    end
  end
end

function Span(el)
  local style = get_custom_style(el.attr)
  if is_text_style(style) then
    return el.content
  end
  return el
end

function Link(el)
  local style = get_custom_style(el.attr)
  if is_text_style(style) then
    return pandoc.Link(el.content, el.target, el.title)
  end
  return el
end

function Image(el)
  if el.attr and el.attr.attributes then
    local new_attributes = {}
    for key, value in pairs(el.attr.attributes) do
      if key ~= "width" and key ~= "height" then
        new_attributes[key] = value
      end
    end
    local new_attr = {el.attr.identifier or "", el.attr.classes or {}, new_attributes}
    return pandoc.Image(el.content or {}, el.src, el.title or "", new_attr)
  end
  return el
end

function Underline(el)
  return el.content
end
