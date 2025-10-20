
-- DOF Headers Filter: Convert Word styles to Markdown headers and clean unwanted attributes

local style_map = {
  ["CABEZA"] = 1,
  ["Titulo 1"] = 2,
  ["Titulo 2"] = 3,
  ["ANOTACION"] = 4,
}

-- Lista de estilos especiales que NO deben tratarse como texto normal
-- Incluye los del style_map más "Title" que se elimina completamente
local special_styles = {
  ["CABEZA"] = true,
  ["Titulo 1"] = true,
  ["Titulo 2"] = true,
  ["ANOTACION"] = true,
  ["Title"] = true,
}

local function get_custom_style(attr)
  return attr and attr.attributes and 
    (attr.attributes["custom-style"] or attr.attributes["custom-style-name"] or attr.attributes["style"])
end

-- Cambio de lógica: ahora verifica si es un estilo ESPECIAL
-- Si no está en special_styles, se trata como texto normal
local function is_text_style(style)
  if not style then return true end  -- Sin estilo = texto normal
  return not special_styles[style]    -- Si no es especial, es texto normal
end

function Div(el)
  local style = get_custom_style(el.attr)
  
  -- Eliminar completamente el contenido si el custom-style es "Title"
  if style == "Title" then
    return {}
  end
  
  -- Si es un estilo de texto normal (no especial), eliminar el div y dejar solo el contenido
  if is_text_style(style) then
    return el.content
  end
  
  -- A partir de aquí, solo procesamos estilos especiales (CABEZA, Titulo 1, Titulo 2, ANOTACION)
  local level = style_map[style]
  if not level then
    -- Si tiene un custom-style pero no está en style_map ni en special_styles,
    -- lo tratamos como texto normal también
    return el.content
  end

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
