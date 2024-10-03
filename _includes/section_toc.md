[Exercise material for this section]({{ site.github.repository_url }}/blob/main/hands-on/{{ page.section }}).

{% for cat in site.menu %}
  {% if cat.name == page.section %}
    {% for item in cat.items %}
- <a href="{{ site.baseurl }}/{{ cat.name }}/{{ item.name }}.html">{% if item.label %}{{ item.label }}{% else %}{{ item.name | capitalize }}{% endif %}</a>
    {% endfor %}
  {% endif %}
{% endfor %}
