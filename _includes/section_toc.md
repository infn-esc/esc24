[Exercise material for this section]({% include hands_on_base.md %}/{{ page.section }}).

{% for cat in site.menu %}
  {% if cat.name == page.section %}
    {% for item in cat.items %}
- <a href="{{ site.baseurl }}/{{ cat.name }}/{{ item.name }}.html">{% if item.label %}{{ item.label }}{% else %}{{ item.name | capitalize }}{% endif %}</a>
    {% endfor %}
  {% endif %}
{% endfor %}
